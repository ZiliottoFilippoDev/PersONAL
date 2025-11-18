import os
import json
import logging
import tkinter as tk
from tkinter import Label, Button, Text, ttk, Entry, PhotoImage
from PIL import Image, ImageTk
import argparse
import threading
from transformers import pipeline  # Import the Hugging Face pipelinea

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ================================
#        MODEL
# ================================

class DataLoader:
    """
    Handles loading of image and JSON data from a dataset directory.
    """

    def __init__(self, root_folder, split="val"):
        """
        Initialize with a root folder and dataset split.
        """
        self.root_folder = root_folder
        self.split = split

    def load_data(self):
        """
        Scans the dataset directory for subfolders containing JSON files and images.

        Returns:
            list: A list of tuples in the format:
                  (list of image paths, object_data, json_file_path, subfolder)
        """
        image_files = []
        data_path = os.path.join(args.root_path, 'data/datasets/eai_pers', self.split)

        if not os.path.exists(data_path):
            logging.error("Data path does not exist: %s", data_path)
            return image_files

        # Iterate over subfolders in the dataset split directory.
        for subfolder in os.listdir(data_path):
            subfolder_path = os.path.join(data_path, subfolder)
            if os.path.isdir(subfolder_path):
                images_path = os.path.join(subfolder_path, "images")
                if not os.path.exists(images_path):
                    logging.warning("Images directory not found in %s", subfolder_path)
                    continue

                # Process each JSON file found in the subfolder.
                for json_file in [f for f in os.listdir(subfolder_path) if f.endswith(".json")]:
                    json_file_path = os.path.join(subfolder_path, json_file)
                    try:
                        with open(json_file_path, "r") as f:
                            data = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logging.error("Error loading JSON file %s: %s", json_file_path, e)
                        continue

                    for obj in data:
                        object_id = obj.get("object_id")
                        if not object_id:
                            logging.warning("Object without object_id in %s", json_file_path)
                            continue

                        # Dynamically find all PNG images matching the pattern: object_id_*.png.
                        image_paths = []
                        for filename in os.listdir(images_path):
                            if filename.startswith(f"{object_id}_") and filename.endswith(".png"):
                                full_path = os.path.join(images_path, filename)
                                image_paths.append(full_path)
                        image_paths = sorted(image_paths)

                        if image_paths:
                            image_files.append((image_paths, obj, json_file_path, subfolder))
                        else:
                            logging.warning("No images found for object_id %s in %s",
                                            object_id, images_path)
        return image_files


class DataModel:
    """
    Represents the data model for the application.
    """

    def __init__(self, root_folder, split="val"):
        self.data_loader = DataLoader(root_folder, split)
        self.items = self.data_loader.load_data()
        self.current_index = 0

    def get_current_item(self):
        """
        Retrieves the current item (with circular navigation).

        Returns:
            tuple: (list of image paths, object_data, json_file_path, subfolder)
        """
        if not self.items:
            return None

        # Circular navigation
        if self.current_index >= len(self.items):
            self.current_index = 0
        elif self.current_index < 0:
            self.current_index = len(self.items) - 1

        return self.items[self.current_index]

    def update_current_item(self, new_item):
        """
        Updates the current item with new data.
        """
        self.items[self.current_index] = new_item

    def next_item(self):
        """
        Move to the next item.
        """
        self.current_index += 1

    def previous_item(self):
        """
        Move to the previous item.
        """
        self.current_index -= 1

# ================================
#         VIEW
# ================================

class ViewerView:
    """
    The GUI view of the application.
    """

    def __init__(self, root, theme="clam", image_size=(300, 300), use_captioner=False):
        self.root = root
        self.image_size = image_size
        self.use_captioner = use_captioner

        self.root.title("Object Viewer")
        self.style = ttk.Style()
        self.style.theme_use(theme)

        # Build the UI layout
        self._create_main_frames()
        self._create_left_panel()
        self._create_right_panel()
        self._create_bottom_panel()

    def _create_main_frames(self):
        self.content_frame = tk.Frame(self.root, padx=10, pady=10)
        self.content_frame.pack(fill=tk.BOTH, expand=True)


    def check_content(self, idx):
        content = self.description_texts[idx].get("1.0", "end-1c").strip()
        if content:
            self.check_labels[idx].configure(image=self.check_img)
        else:
            self.check_labels[idx].configure(image=self.empty_img)
    
    
    def _create_left_panel(self):
        self.text_frame = tk.Frame(self.content_frame, padx=10, pady=10,
                                   relief=tk.GROOVE, borderwidth=2)
        self.text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Folder location
        self.folder_label = Label(self.text_frame, text="Folder Location:",
                                  font=("Arial", 12, "bold"))
        self.folder_label.pack(anchor="w")
        self.folder_value = Label(self.text_frame, text="",
                                  font=("Arial", 12), wraplength=300)
        self.folder_value.pack(anchor="w", pady=5)

        # Category
        self.category_frame = tk.Frame(self.text_frame)
        self.category_frame.pack(anchor="w", pady=5)
        self.category_label = Label(self.category_frame, text="Category:",
                                    font=("Arial", 12, "bold"))
        self.category_label.pack(side=tk.LEFT)
        self.category_value = Label(self.category_frame, text="",
                                    font=("Arial", 12), wraplength=300)
        self.category_value.pack(side=tk.LEFT, padx=(5, 0))

        # Object ID
        self.object_id_frame = tk.Frame(self.text_frame)
        self.object_id_frame.pack(anchor="w", pady=5)
        self.object_id_label = Label(self.object_id_frame, text="Object ID:",
                                     font=("Arial", 12, "bold"))
        self.object_id_label.pack(side=tk.LEFT)
        self.object_id_value = Label(self.object_id_frame, text="",
                                     font=("Arial", 12), wraplength=300)
        self.object_id_value.pack(side=tk.LEFT, padx=(5, 0))

        # Floor ID
        self.floor_id_frame = tk.Frame(self.text_frame)
        self.floor_id_frame.pack(anchor="w", pady=5)
        self.floor_id_label = Label(self.floor_id_frame, text="Floor ID:",
                                    font=("Arial", 12, "bold"))
        self.floor_id_label.pack(side=tk.LEFT)

        self.floor_id_var = tk.StringVar(value="0")  # default: 0
          # callback trace che si attiva quando cambia il valore
        self.floor_id_var.trace_add("write", self.on_floor_id_change)
        self.floor_id_menu = tk.OptionMenu(self.floor_id_frame, self.floor_id_var, "0", "1", "2")
        self.floor_id_menu.config(font=("Arial", 12))

        # self.floor_id_value = Label(self.floor_id_frame, text="",
                                    # font=("Arial", 12), wraplength=300)
        # self.floor_id_value.pack(side=tk.LEFT, padx=(5, 0))
        self.floor_id_menu.pack(side=tk.LEFT, padx=(5, 0))

        # Load green check image
        original_img = PhotoImage(file="./assets/check_icon.png") #cambiare in assets

        self.check_img = original_img.subsample(20, 20)
        self.empty_img = PhotoImage()  # immagine vuota per placeholder

        self.description_labels = []
        self.description_texts = []
        self.check_labels = []

        for i in range(3):

            # Frame to contain "Description" and green check
            label_frame = tk.Frame(self.text_frame)
            label_frame.pack(anchor="w") 

            label = Label(label_frame, text=f"Description {i+1}:", font=("Arial", 12, "bold"))
            # label.pack(anchor="w")
            label.pack(side="left")
            self.description_labels.append(label)

            check = Label(label_frame, image=self.empty_img)
            check.pack(side="left", padx=5)
            self.check_labels.append(check)


            text_widget = Text(self.text_frame, height=3, width=40, font=("Arial", 12))
            text_widget.pack(anchor="w", pady=5)
            # Track modifications to load green check
            text_widget.bind("<KeyRelease>", lambda e, idx=i: self.check_content(idx))
            self.description_texts.append(text_widget)

            self.check_content(i)
        
        # Room Name
        self.room_label = Label(self.text_frame, text="Room Name:",
                                font=("Arial", 12, "bold"))
        self.room_label.pack(anchor="w")
        self.room_text = Text(self.text_frame, height=1, width=40,
                              font=("Arial", 12))
        self.room_text.pack(anchor="w", pady=5)

        # Button toolbar
        self.button_frame = tk.Frame(self.text_frame, padx=10, pady=5,
                                     relief=tk.RAISED, borderwidth=2)
        self.button_frame.pack(anchor="w", pady=5, fill=tk.X)

        self.back_button = Button(self.button_frame, text="Back")
        self.back_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.next_button = Button(self.button_frame, text="Next")
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.refine_button = Button(self.button_frame, text="Refine")
        self.refine_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.delete_frame = tk.Frame(self.text_frame, padx=10, pady=5,
                                     relief=tk.RAISED, borderwidth=2)
        self.delete_frame.pack(fill=tk.X, pady=10)

        self.delete_button = Button(self.delete_frame, text="Delete", bg="red", fg="black", height=1, width=10)
        self.delete_button.pack(side=tk.TOP, padx=10, pady=5)

        #Borderline button
        self.borderline_var = tk.BooleanVar()
        self.borderline_checkbox_frame = tk.Frame(self.text_frame, padx=15, pady=5,
                                     relief=tk.RAISED, borderwidth=2)
        self.borderline_checkbox_frame.pack(fill=tk.X, pady=15)

        self.borderline_checkbox = tk.Checkbutton(self.borderline_checkbox_frame, text="Borderline", variable=self.borderline_var)
        self.borderline_checkbox.pack(side=tk.LEFT, padx=100)

        # New Automatic Description button, only if use_captioner is True
        self.auto_desc_button = Button(self.button_frame, text="Automatic Description")
        if self.use_captioner:
            self.auto_desc_button.pack(side=tk.LEFT, padx=5, pady=5)
        else:
            self.auto_desc_button.pack_forget()

    def _create_right_panel(self):
        self.image_frame = tk.Frame(self.content_frame, padx=10, pady=10,
                                    relief=tk.GROOVE, borderwidth=2)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_bottom_panel(self):
        self.status_frame = tk.Frame(self.root, relief=tk.SUNKEN, borderwidth=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = Label(self.status_frame, text="",
                                  font=("Arial", 10), anchor="w")
        self.status_label.pack(fill=tk.X, padx=5, pady=2)

        self.progress_frame = tk.Frame(self.root, padx=10, pady=5)
        self.progress_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, orient="horizontal", length=200, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X)

    def update_text_fields(self, obj, subfolder, split):
        """
        Update the text fields based on the given object data.
        """
        self.category_value.config(text=obj.get("object_category", ""))
        self.object_id_value.config(text=obj.get("object_id", ""))
        # self.floor_id_value.config(text=obj.get("floor_id", ""))
        self.floor_id_var.set(obj.get("floor_id", ""))
        self.folder_value.config(text=f"{split}/{subfolder}")
        # -------------------------------
        # Modified: Update three description fields
        # -------------------------------
        descriptions = obj.get("description", ["", "", ""])
        if not isinstance(descriptions, list) or len(descriptions) < 3:
            descriptions = descriptions if isinstance(descriptions, list) else [descriptions]
            descriptions = (descriptions + [""] * 3)[:3]
        for i, text_widget in enumerate(self.description_texts):
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, descriptions[i])
        
        self.room_text.delete(1.0, tk.END)
        self.room_text.insert(tk.END, obj.get("room", ""))

    def update_status(self, index, total):
        """
        Update the status label and progress bar.
        """
        self.progress_bar.config(value=index + 1, maximum=total)
        self.status_label.config(text=f"Item {index + 1} of {total}")

    def clear_images(self):
        """
        Clear any images currently displayed.
        """
        for widget in self.image_frame.winfo_children():
            widget.destroy()

    def display_images(self, image_paths):
        """
        Display images in a grid format in the image frame.
        """
        container = tk.Frame(self.image_frame)
        container.pack()
        cols = 2
        for idx, path in enumerate(image_paths):
            try:
                image = Image.open(path)
                image = image.resize(self.image_size, Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
            except Exception as e:
                logging.error("Error loading image %s: %s", path, e)
                continue

            img_label = Label(container, image=photo)
            img_label.image = photo  # Prevent garbage collection.
            row, col = divmod(idx, cols)
            img_label.grid(row=row, column=col, padx=5, pady=5)

    def load_borderline(self, obj):
        """Set checkbox state according to value to_discuss in json file"""
        to_discuss = obj.get("to_discuss", False)
        
        # Converti vari possibili valori a booleano
        if isinstance(to_discuss, str):
            to_discuss = to_discuss.lower() in ("true")
        elif isinstance(to_discuss, (int, float)):
            to_discuss = bool(to_discuss)
        else:
            to_discuss = bool(to_discuss)
            
        self.borderline_var.set(to_discuss)
    
    def on_floor_id_change(self, *args):
        """Chiamato ogni volta che cambia il valore del menu a tendina floor_id"""
        # Implementato con un callback nel controller
        if hasattr(self, 'floor_id_callback') and callable(self.floor_id_callback):
            self.floor_id_callback()


# ================================
#      CONTROLLER
# ================================

class ViewerController:
    def __init__(self, root_folder, split="val_unseen", theme="clam", image_size=(300, 300), use_captioner=False):
        self.model = DataModel(root_folder, split)
        self.split = split
        self.use_captioner = use_captioner
        self.view = ViewerView(tk.Tk(), theme, image_size, use_captioner=self.use_captioner)

         # Riferimento al metodo del controller da chiamare quando cambia floor_id
        self.view.floor_id_callback = self.on_floor_id_change
        
        # Bind buttons to controller methods.
        self.view.back_button.config(command=self.previous_item)
        self.view.next_button.config(command=self.next_item)
        self.view.refine_button.config(command=self.refine_description)
        self.view.delete_button.config(command=self.delete_item)
        if self.use_captioner:
            self.view.auto_desc_button.config(command=self.auto_generate_description)

        self.view.borderline_checkbox.config(command=self.on_borderline_toggle)
       
        # Add entry and button for jumping to an object/item
        self.view.jump_frame = tk.Frame(self.view.root, padx=10, pady=5)
        self.view.jump_frame.pack(side=tk.TOP, fill=tk.X)
        self.view.jump_entry = Entry(self.view.jump_frame, font=("Arial", 12))
        self.view.jump_entry.pack(side=tk.LEFT, padx=5)
        self.view.jump_button = Button(self.view.jump_frame, text="Go to", command=self.jump_to_item)
        self.view.jump_button.pack(side=tk.LEFT, padx=5)
        
        # Bind keyboard shortcuts
        self.view.root.bind("<Left>", lambda event: self.previous_item())
        self.view.root.bind("<Right>", lambda event: self.next_item())
        self.view.root.bind("<Up>", lambda event: self.next_item())
        self.view.root.bind("<Down>", lambda event: self.previous_item())
        self.view.root.bind("<Return>", lambda event: self.refine_description())
        self.view.root.bind("<Control-s>", lambda event: self.refine_description())
        self.view.root.bind("<Command-s>", lambda event: self.refine_description())
        self.view.root.bind("<Command-b>", lambda event: self.toggle_borderline_shortcut())
        self.view.root.bind("<Command-d>", lambda event: self.delete_item())
        # self.view.root.bind("b", lambda event: self.toggle_borderline_shortcut())
        # self.view.root.bind("d", lambda event: self.delete_item())
        
        if not self.model.items:
            logging.error("No data loaded. Exiting application.")
            self.view.root.destroy()
            return

        # Initialize the Hugging Face image captioning pipeline once if enabled.
        if self.use_captioner:
            # model_name = "nlpconnect/vit-gpt2-image-captioning"
            model_name = "Salesforce/blip-image-captioning-large"
            self.captioner = pipeline("image-to-text", model=model_name)
        else:
            self.captioner = None

        self.load_current_item()
        self.view.root.mainloop()


    def on_borderline_toggle(self):
        """Update to_discuss field in json file when checkbox is toggled"""
        item = self.model.get_current_item()
        if not item:
            return
            
        image_paths, obj, json_file_path, subfolder = item
        is_borderline = bool(self.view.borderline_var.get())  # Forza a booleano
        
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
                
            for item_obj in data:
                if item_obj.get("object_id") == obj.get("object_id"):
                    item_obj["to_discuss"] = is_borderline  # Salva come booleano
                    obj = item_obj
                    break
                    
            with open(json_file_path, "w") as f:
                # Assicura che i booleani vengano serializzati come veri valori booleani
                json.dump(data, f, indent=4, default=lambda x: x)
                
            self.model.update_current_item((image_paths, obj, json_file_path, subfolder))
            
        except Exception as e:
            logging.error(f"Error updating to_discuss status: {e}")

    def load_current_item(self):
        item = self.model.get_current_item()
        if not item:
            return

        image_paths, obj, _, subfolder = item
        total = len(self.model.items)

        self.view.update_text_fields(obj, subfolder, self.split)
        self.view.update_status(self.model.current_index, total)

        # Set checkbox borderline state
        self.view.load_borderline(obj)
        
        self.view.clear_images()
        self.view.display_images(image_paths)

        #########
        # Controlla se le descrizioni sono già precompilate e aggiorna le spunte
        for idx, text_widget in enumerate(self.view.description_texts):
            if text_widget.get(1.0, tk.END).strip():  # Se c'è già del testo
                self.view.check_labels[idx].config(image=self.view.check_img)  # Aggiungi la spunta
            else:
                self.view.check_labels[idx].config(image=self.view.empty_img)  # Rimuovi la spunta
        #########

    def next_item(self):
        self.model.next_item()
        self.load_current_item()
    
    def previous_item(self):
        self.model.previous_item()
        self.load_current_item()
    
    def toggle_borderline_shortcut(self):
        """Toggle the borderline checkbox when 'b' is pressed"""
        current_state = self.view.borderline_var.get()
        self.view.borderline_var.set(not current_state)  # Inverte lo stato
    
        # Chiama manualmente la funzione di toggle per aggiornare il file JSON
        self.on_borderline_toggle()
            
    def refine_description(self):
        item = self.model.get_current_item()
        if not item:
            return

        image_paths, obj, json_file_path, subfolder = item

        new_descriptions = [text_widget.get(1.0, tk.END).strip() for text_widget in self.view.description_texts]
        new_room = self.view.room_text.get(1.0, tk.END).strip()
        new_floor_id = self.view.floor_id_var.get()  # Ottieni il floor_id dal menu a tendina

        # Get borderline state
        is_borderline = self.view.borderline_var.get()

        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error("Error reading JSON file %s: %s", json_file_path, e)
            return

        updated = False
        for item_obj in data:
            if item_obj.get("object_id") == obj.get("object_id"):
                item_obj["description"] = new_descriptions
                item_obj["room"] = new_room
                item_obj["to_discuss"] = is_borderline
                item_obj["floor_id"] = new_floor_id  # Salva il floor_id
                updated = True
                obj = item_obj
                break

        if not updated:
            logging.warning("Object with ID %s not found in %s", obj.get("object_id"), json_file_path)
            return

        try:
            with open(json_file_path, "w") as f:
                json.dump(data, f, indent=4)
            logging.info("Updated object ID %s in %s", obj.get("object_id"), json_file_path)
        except Exception as e:
            logging.error("Error writing JSON file %s: %s", json_file_path, e)
            return

        self.model.update_current_item((image_paths, obj, json_file_path, subfolder))

        for idx, text_widget in enumerate(self.view.description_texts):
            if text_widget.get(1.0, tk.END).strip():  # Se c'è già del testo
                self.view.check_labels[idx].config(image=self.view.check_img)  # Aggiungi la spunta
            else:
                self.view.check_labels[idx].config(image=self.view.empty_img)  # Rimuovi la spunta

        self.load_current_item()
    
    def jump_to_item(self):
        input_value = self.view.jump_entry.get().strip()
        
        if input_value.isdigit():
            index = int(input_value) - 1
            if 0 <= index < len(self.model.items):
                self.model.current_index = index
                self.load_current_item()
            else:
                logging.warning("Index out of range: %s", input_value)
        else:
            for idx, (image_paths, obj, _, _) in enumerate(self.model.items):
                if obj.get("object_id") == input_value:
                    self.model.current_index = idx
                    self.load_current_item()
                    return
            logging.warning("Object ID not found: %s", input_value)

    def auto_generate_description(self):
        """
        Uses the Hugging Face image captioning pipeline to generate a description for the first displayed image.
        """
        if not self.captioner:
            logging.warning("Captioner not enabled.")
            return

        item = self.model.get_current_item()
        if not item:
            return

        image_paths, obj, json_file_path, subfolder = item
        
        # Choose the upper right image if available, otherwise the first one.
        caption_image_path = image_paths[1] if len(image_paths) >= 2 else image_paths[0]
        
        def run_caption():
            try:
                result = self.captioner(caption_image_path, max_new_tokens=50)
                caption = result[0]['generated_text']
            except Exception as e:
                caption = ""
                logging.error("Caption generation error: %s", e)
            
            # Safely update the GUI from the main thread.
            self.view.root.after(0, lambda: self.view.description_texts[0].delete(1.0, tk.END))
            self.view.root.after(0, lambda: self.view.description_texts[0].insert(tk.END, caption))
        
        # Run the captioning in a background thread.
        threading.Thread(target=run_caption).start()

    # Mostra un popup di conferma e se confermato, elimina l'elemento corrente.
    def delete_item(self):

        item = self.model.get_current_item()
        if not item:
            return

        # Crea una finestra di dialogo di conferma
        confirm_dialog = tk.Toplevel(self.view.root)
        confirm_dialog.title("Conferma eliminazione")
        confirm_dialog.geometry("300x150")
        confirm_dialog.resizable(False, False)
        
        # Rendi la finestra modale (blocca l'interazione con la finestra principale)
        confirm_dialog.transient(self.view.root)
        confirm_dialog.grab_set()
        
        # Centra la finestra rispetto alla finestra principale
        x = self.view.root.winfo_x() + self.view.root.winfo_width()//2 - 150
        y = self.view.root.winfo_y() + self.view.root.winfo_height()//2 - 75
        confirm_dialog.geometry(f"+{x}+{y}")
        
        # Aggiungi messaggio di conferma
        message = Label(confirm_dialog, text="Sei sicuro di voler eliminare questo elemento?", 
                        font=("Arial", 12), wraplength=280, pady=15)
        message.pack()
        
        # Frame per i pulsanti
        button_frame = tk.Frame(confirm_dialog)
        button_frame.pack(pady=10)
        
        # Funzione per eliminare effettivamente l'elemento
        def confirm_delete():
            
            image_paths, obj, json_file_path, subfolder = item
            object_id = obj.get("object_id", "unknown")
            logging.info(image_paths)

            # Trova l'indice corrente e determina l'id dell'oggetto successivo o precedente
            current_idx = self.model.current_index
            next_idx = min(current_idx + 1, len(self.model.items) - 1)
    
            # Salva l'ID dell'oggetto successivo o precedente per riposizionarsi dopo
            next_obj_id = None
            if next_idx < len(self.model.items):
                next_obj_id = self.model.items[next_idx][1].get("object_id")

            # Elimina le immagini associate all'oggetto
            for img_path in image_paths:
                try:
                    os.remove(img_path)
                    logging.info(f"Eliminata immagine: {img_path}")
                except Exception as e:
                    logging.error(f"Errore nell'eliminazione dell'immagine {img_path}: {e}")
            
            try:
                # Carica i dati dal file JSON
                with open(json_file_path, "r") as f:
                    data = json.load(f)

                    # Rimuove l'oggetto dal JSON
                    data = [item_obj for item_obj in data if item_obj.get("object_id") != object_id]


                # Salva il JSON aggiornato
                with open(json_file_path, "w") as f:
                    json.dump(data, f, indent=4)
                    
                logging.info(f"Eliminazione confermata per l'oggetto {object_id}")
            
            except Exception as e:
                logging.error(f"Errore nell'aggiornamento del JSON {json_file_path}: {e}")
    
            
            
            # Chiudi il dialogo
            confirm_dialog.destroy()
    
            # Ricarica i dati del modello poiché ora sono cambiati
            self.model = DataModel(self.model.data_loader.root_folder, self.split)

            # Riposizionati all'oggetto successivo se disponibile
            if next_obj_id:
                for idx, (_, obj, _, _) in enumerate(self.model.items):
                    if obj.get("object_id") == next_obj_id:
                        self.model.current_index = idx
                        break
            
            # Carica l'elemento corrente
            self.load_current_item()

        
        # Funzione per annullare l'eliminazione
        def cancel_delete():
            confirm_dialog.destroy()
        
        # Pulsanti Sì e No
        yes_button = Button(button_frame, text="Sì", width=8, command=confirm_delete)
        yes_button.pack(side=tk.LEFT, padx=10)
        
        no_button = Button(button_frame, text="No", width=8, command=cancel_delete)
        no_button.pack(side=tk.LEFT, padx=10)
        
        # Imposta il focus sul pulsante No come precauzione
        no_button.focus_set()

    def on_floor_id_change(self):
        """Gestisce il cambiamento del floor_id e lo salva nel JSON"""
        item = self.model.get_current_item()
        if not item:
            return
            
        image_paths, obj, json_file_path, subfolder = item
        new_floor_id = self.view.floor_id_var.get()
        
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
                
            for item_obj in data:
                if item_obj.get("object_id") == obj.get("object_id"):
                    item_obj["floor_id"] = new_floor_id
                    obj = item_obj  # Aggiorna anche l'oggetto locale
                    break
                    
            with open(json_file_path, "w") as f:
                json.dump(data, f, indent=4)
                
            # Aggiorna l'oggetto nel modello
            self.model.update_current_item((image_paths, obj, json_file_path, subfolder))
            logging.info(f"Aggiornato floor_id a {new_floor_id} per l'oggetto {obj.get('object_id')}")
            
        except Exception as e:
            logging.error(f"Errore nell'aggiornamento del floor_id: {e}")

# ================================
#             MAIN
# ================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Object Viewer for EAI Dataset")
    parser.add_argument("--split", type=str, default="val_unseen",
                        help="Dataset split to use (e.g., 'val_unseen', 'train')")
    parser.add_argument("--theme", type=str, default="classic",
                        help="Theme to use for the GUI (e.g., 'clam', 'classic')")
    parser.add_argument("--root_path", type=str, default="",
                        help="Root folder containing the dataset")
    parser.add_argument("--use_captioner", action="store_true", default=False,
                        help="If provided, enable the automatic captioner functionality")
    args = parser.parse_args()
    
    # Set the root folder path (adjust as needed)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    args.root_path = os.path.abspath(os.path.join(current_dir, "../.."))

    assert args.split in ["val_unseen", "val_seen", "val_seen_synonyms", "train"], "Invalid split value. Use 'val_unseen' or 'train'."
    assert args.root_path, "Root folder path is required."

    ViewerController(args.root_path, split=args.split, theme=args.theme, use_captioner=args.use_captioner)
