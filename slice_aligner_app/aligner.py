import tkinter as tk
from tkinter import filedialog, messagebox, Scale
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json

class ManualAligner:
    """
    A tkinter application for manually aligning a 'moving' image to a 'reference' image.

    The user can load two images and use sliders to adjust the rotation and
    X/Y translation of the moving image. An opacity slider helps in visualizing
    the alignment. The resulting transformation parameters can be saved to a JSON file.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Image Aligner")
        self.root.geometry("1200x800")

        # --- State Variables ---
        self.ref_image_path = None
        self.mov_image_path = None
        self.ref_image_cv = None
        self.mov_image_cv = None
        self.display_image = None
        self.canvas_image_id = None

        # --- Transformation Parameters ---
        self.angle = tk.DoubleVar()
        self.dx = tk.DoubleVar()
        self.dy = tk.DoubleVar()
        self.opacity = tk.DoubleVar(value=0.5)

        # --- GUI Layout ---
        # Main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame on the left
        control_frame = tk.Frame(main_frame, width=250, bd=2, relief=tk.RIDGE)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Canvas for image display on the right
        self.canvas = tk.Canvas(main_frame, bg='gray')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Controls ---
        # File buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(pady=10, padx=5, fill=tk.X)
        tk.Button(btn_frame, text="Load Reference Image", command=self.load_reference_image).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Load Moving Image", command=self.load_moving_image).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Save Transformation", command=self.save_transformation, bg="#4CAF50", fg="white").pack(fill=tk.X, pady=(15, 5))

        # Sliders
        slider_frame = tk.Frame(control_frame)
        slider_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(slider_frame, text="Rotation (°)").pack()
        self.rot_slider = Scale(slider_frame, from_=-180, to=180, resolution=0.1, orient=tk.HORIZONTAL, variable=self.angle, command=self.update_display)
        self.rot_slider.pack(fill=tk.X)

        tk.Label(slider_frame, text="X-Shift (px)").pack()
        self.dx_slider = Scale(slider_frame, from_=-200, to=200, resolution=1, orient=tk.HORIZONTAL, variable=self.dx, command=self.update_display)
        self.dx_slider.pack(fill=tk.X)

        tk.Label(slider_frame, text="Y-Shift (px)").pack()
        self.dy_slider = Scale(slider_frame, from_=-200, to=200, resolution=1, orient=tk.HORIZONTAL, variable=self.dy, command=self.update_display)
        self.dy_slider.pack(fill=tk.X)

        tk.Label(slider_frame, text="Opacity").pack()
        self.opacity_slider = Scale(slider_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=self.opacity, command=self.update_display)
        self.opacity_slider.pack(fill=tk.X)

    def load_reference_image(self):
        """Load the fixed reference image."""
        path = filedialog.askopenfilename()
        if not path:
            return
        self.ref_image_path = path
        self.ref_image_cv = cv2.imread(self.ref_image_path)
        self.update_display()
        self.root.title(f"Manual Image Aligner - Ref: {os.path.basename(path)}")

    def load_moving_image(self):
        """Load the moving image to be aligned."""
        path = filedialog.askopenfilename()
        if not path:
            return
        self.mov_image_path = path
        self.mov_image_cv = cv2.imread(self.mov_image_path)
        self.reset_sliders()
        self.update_display()

    def reset_sliders(self):
        """Reset sliders to their default positions."""
        self.angle.set(0)
        self.dx.set(0)
        self.dy.set(0)

    def update_display(self, *args):
        """Core function to transform, blend, and display the images."""
        if self.ref_image_cv is None:
            return

        # Start with the reference image as the base layer
        display_cv = self.ref_image_cv.copy()

        if self.mov_image_cv is not None:
            h_ref, w_ref, _ = display_cv.shape
            h_mov, w_mov, _ = self.mov_image_cv.shape

            # --- FIX: Ensure the background canvas is large enough for both images ---
            if h_mov > h_ref or w_mov > w_ref:
                # Create a new, larger canvas based on the max dimensions
                max_h = max(h_ref, h_mov)
                max_w = max(w_ref, w_mov)
                
                new_display_cv = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                
                # Place the original, smaller reference image in the center of the new canvas
                y_offset = (max_h - h_ref) // 2
                x_offset = (max_w - w_ref) // 2
                new_display_cv[y_offset:y_offset+h_ref, x_offset:x_offset+w_ref] = display_cv
                
                # This is now our base layer for blending
                display_cv = new_display_cv
            
            # Now, display_cv is guaranteed to be large enough to hold mov_image_cv.
            # Update h_ref and w_ref to the current canvas size for transformation calculations.
            h_ref, w_ref, _ = display_cv.shape
            # --- END FIX ---

            # Create a transparent overlay for the moving image, correctly centered
            padded_mov = np.zeros_like(display_cv, dtype=np.uint8)
            y_offset = (h_ref - h_mov) // 2
            x_offset = (w_ref - w_mov) // 2
            
            # This assignment is now safe because display_cv is guaranteed to be large enough
            padded_mov[y_offset:y_offset+h_mov, x_offset:x_offset+w_mov] = self.mov_image_cv

            # 1. Get transformation parameters from sliders
            angle = self.angle.get()
            dx = self.dx.get()
            dy = self.dy.get()

            # 2. Construct the affine transformation matrix
            center = (w_ref / 2, h_ref / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += dx
            M[1, 2] += dy

            # 3. Apply the transformation
            warped_mov = cv2.warpAffine(padded_mov, M, (w_ref, h_ref))

            # 4. Blend the images
            opacity = self.opacity.get()
            display_cv = cv2.addWeighted(warped_mov, opacity, display_cv, 1 - opacity, 0)

        # Convert from OpenCV BGR to PIL RGB format, then to Tkinter PhotoImage
        img_rgb = cv2.cvtColor(display_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.display_image = ImageTk.PhotoImage(image=img_pil)

        # Display on canvas
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

    def save_transformation(self):
        """Save the current transformation parameters to a JSON file."""
        if not self.mov_image_path:
            messagebox.showerror("Error", "No moving image is loaded.")
            return

        transform_data = {
            "reference_image": self.ref_image_path,
            "moving_image": self.mov_image_path,
            "angle_degrees": self.angle.get(),
            "dx_pixels": self.dx.get(),
            "dy_pixels": self.dy.get()
        }

        # Create the output filename
        directory, filename = os.path.split(self.mov_image_path)
        name, _ = os.path.splitext(filename)
        output_filename = os.path.join(directory, f"{name}_transform.json")

        try:
            with open(output_filename, 'w') as f:
                json.dump(transform_data, f, indent=4)
            messagebox.showinfo("Success", f"Transformation saved to:\n{output_filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ManualAligner(root)
    root.mainloop()

