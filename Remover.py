from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
from PIL import Image, ImageFilter
from rembg import remove
import numpy as np
import torch
import cv2
import dlib
import os

class Remover:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "apply_effect": (["Remove Background", "Remove Face", "Blur"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "Remover"

    def main(self, image, apply_effect):
        if image.dim() == 4:
            image = image.squeeze(0).permute(2, 0, 1)  # Remove batch dimension and permute

        img = ToPILImage()(image) # Convert the tensor to a PIL Image

        if apply_effect == "Remove Background":
            img = self.remove_background(img)
        elif apply_effect == "Remove Face":
            img = self.remove_face(img)
        elif apply_effect == "Blur":
            img = self.blur_image(img)
        
        processed_tensor = ToTensor()(img) # Convert the PIL Image back to a tensor
        processed_tensor = processed_tensor.unsqueeze(0).permute(0, 2, 3, 1) # Put back the batch dimension and permute back to [batch_size, height, width, channels]
        return processed_tensor,

    def remove_face(self, image):
        image = image.convert("RGBA")
        image_rgb = image.convert("RGB") # Convert image to RGB 
        image_cv = np.array(image_rgb)
        image_cv = image_cv[:, :, ::-1].copy()  # Convert RGB to BGR (OpenCV works with BGR)
        detector = dlib.get_frontal_face_detector() #Init detector
        predictor = dlib.shape_predictor(os.getcwd() + "/custom_nodes/shape_predictor_68_face_landmarks.dat")
        faces = detector(image_cv, 1) # Detect faces
        mask = np.zeros_like(image_cv) # Create a mask with the same dimensions as the image

        # For each face detected
        for face in faces:
            landmarks = predictor(image_cv, face)
            face_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)], dtype=np.int32) # Get points for the entire face
            topmost_eyebrow = min(face_points[17:27], key=lambda x: x[1])  # Points from the eyebrows
            chin_point = face_points[8]  # Chin points
            forehead_height_estimate = int(abs(chin_point[1] - topmost_eyebrow[1]) * 0.5) # Estimate the forehead height as a proportion of the face height
            forehead_point = (topmost_eyebrow[0], topmost_eyebrow[1] - forehead_height_estimate)
            extended_face_points = np.append(face_points, [forehead_point], axis=0) # Add the estimated forehead point to the face points
            hull = cv2.convexHull(extended_face_points) # Create a convex hull around the extended face points
            cv2.fillConvexPoly(mask, hull, (255, 255, 255)) # Fill the convex hull with white

        alpha_mask = mask[:, :, 0] # Extract alpha channel from the mask and invert it
        alpha_mask_inv = cv2.bitwise_not(alpha_mask)
        r, g, b, a = cv2.split(np.array(image)) # Split the image into colors and alpha 
        final_a = cv2.bitwise_and(a, alpha_mask_inv) # invert alpha mask
        final_image = cv2.merge([r, g, b, final_a]) # Combine the color channels and alpha
        result_image = Image.fromarray(final_image, 'RGBA')

        return result_image
    
    def blur_image(self, img, blur=5):    
        return img.filter(ImageFilter.GaussianBlur(blur))

    def remove_background(self, img):
        return remove(img)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Remover": Remover
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Remover": "Remove Parts"
}
