# Python script to save descriptions from data/datasets/eai_pers/val_[seen/unseen/synonyms] to
# data/datasets/goat_bench/hm3d/v3

import argparse
import os
import json
import gzip
import shutil
# from collections import defaultdict


def main():


    def save_description(scene, object_id, description, room):

        scene_file_path = os.path.join(output_path, f"{scene}.json.gz")

        # File temporaneo per la scrittura, così nel caso succedessero problemi durante la scrittura non
        # compromettiamo il file originale
        temp_file_path = scene_file_path + ".temp"
                    
        try:
            # Leggere il file compresso
            with gzip.open(scene_file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                # print("    " + json.dumps(data, indent=2).replace('\n', '\n    '))

            # Flag per tracciare se ci sono stati aggiornamenti
            updates_made = False
                
            # Aggiornare le descrizioni
            # struttura di data.items(): ('BAbdmeyTvMZ.basis.glb_picture', [{'object_category': 'picture', 'object_id': etc....)
            for obj_key, obj_list in data.items():
                for obj in obj_list: #scorro sulle chiavi
                    if "object_id" and obj["object_id"] == object_id:
                        obj["lang_desc"] = description
                        obj["room"] = room
                        updates_made = True
                        print(f"Aggiornato object_id: {obj_key} - {object_id}")

            # Scrivere i dati aggiornati se ci sono modifiche
            if updates_made:
                with gzip.open(temp_file_path, 'wt', encoding='utf-8') as f:
                     json.dump(data, f, indent=2)
                
                # Sostituire il file originale con quello aggiornato
                shutil.move(temp_file_path, scene_file_path)
                print(f"File aggiornato con successo.")

                # Rimuovere il file temporaneo se esiste
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        except Exception as e:
            print(f"    Errore nell'aggiornamento del file {scene}.json.gz: {str(e)}")
            
            # Rimuovere il file temporaneo in caso di errore
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    parser = argparse.ArgumentParser(description="Folder for save description")
    parser.add_argument("--split", type=str, default="val_unseen",
                        help="Dataset split to use (e.g., 'val_unseen', 'val_synonyms')")
    
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(current_dir, "../../data/datasets/"))
    
    val_path = os.path.join(data_path, "eai_pers/", args.split)

    output_path = os.path.join(data_path, f"goat_bench/hm3d/v3/{args.split}/content/")

    if os.path.exists(val_path):
        items = os.listdir(val_path)
        scenes = []
        
        # Filtra solo le directory
        for item in items:
            item_path = os.path.join(val_path, item)
            if os.path.isdir(item_path):
                scenes.append(item)

        
        if scenes:
                for scene in scenes:
                    print(f"- Scene: {scene}")
                    files_path = os.path.join(val_path, scene)

                    try:
                        files = os.listdir(files_path)
                        
                        print(f"  File JSON nella scena {scene}:")
                        for file in files:
                            if file.endswith('.json'):
                                file_path = os.path.join(files_path, file)
                                print(f"  - {file}")
                                
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        json_content = json.load(f)
                                        # print(json.dumps(json_content, indent=2))
                                        for obj in json_content:
                                                if "object_id" in obj:
                                                    # print(f"object_id: {obj['object_id']}")
                                                    object_id = obj['object_id']

                                                    # Per ora prende solo description 1 (valutare se usare le liste per passarle tutte)
                                                    description = obj["description"][0]
                                                    room = obj["room"]
                                                    save_description(scene, object_id, description, room)
                                except json.JSONDecodeError:
                                    print(f"    Errore: Il file {file} non contiene JSON valido.")
                                except Exception as e:
                                    print(f"    Errore nella lettura del file {file}: {str(e)}")
                    except Exception as e:
                        print(f"  Errore nell'accesso alla scena {scene}: {str(e)}")
        else:
            print("Nessuna directory (scena) trovata.")
    else:
        print(f"La directory {val_path} non esiste.")
                    

if __name__ == "__main__":
    main()