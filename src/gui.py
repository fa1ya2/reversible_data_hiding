import tkinter as tk
from PIL import ImageTk, Image
from src.data_extraction import extract_data

class GUI:
    def __init__(self, image_path, extracted_text):
        self.window = tk.Tk()
        self.window.title("Embedded Text Viewer")

        # Load the image
        image = Image.open(image_path)
        image = image.resize((500, 500), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image)

        # Create the image widget
        self.image_widget = tk.Label(self.window, image=self.photo)
        self.image_widget.pack()

        # Create a tooltip label
        self.tooltip = tk.Label(self.window, text="", bg="white", relief="solid", anchor="w")

        # Bind the hover event to the image widget
        self.image_widget.bind("<Enter>", self.show_text)
        self.image_widget.bind("<Leave>", self.hide_text)

        # Store the extracted text
        self.extracted_text = extracted_text

    def show_text(self, event):
        # Display the extracted text on hover
        self.tooltip.config(text=self.extracted_text)
        self.tooltip.place(x=event.x, y=event.y + 20)

    def hide_text(self, event):
        # Hide the tooltip when mouse leaves the image
        self.tooltip.place_forget()

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    image_path = "output/embedded_image.png"
    extracted_text_path = "output/extracted_data_decrypted.txt"

    with open(extracted_text_path, "r") as file:
        extracted_text = file.read()

    gui = GUI(image_path, extracted_text)
    gui.run()
