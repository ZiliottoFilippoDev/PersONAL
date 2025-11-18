# Annotation Rules

This guide outlines the rules for proper annotation to ensure consistency and clarity.

## Descriptions
- Each object should have **2 descriptions** (3 if time permits, but 2 are sufficient for now).
- Rooms are not always mandatory, especially for objects that cannot really be 
- Descriptions must be **at most 3 rows in length** in the annotator.
- If an object already has an annotation, clean it and rewrite it if it is non-informative.
- The best possible description should include:
  - The **color** of the object.
  - Recognizable **color patterns**
  - The **room location** where the object is found.
  - The **spatial relation** to other objects in the scene.
- A description is considered sufficient if it includes at least **2 of the attributes** described above. If onyl the **spatial relation** is available then we make sure it is described in detail.
- If no attribute can be described, the pictures are considered non-informative, and the object should be deleted.

## Rooms
- We annotate the room, only if we are sure that the room is correct.
- The possibile room list to chose are:
    - Living Room / TV room
    - Kitchen
    - Bedroom
    - Bathroom
    - Dining Room
    - Office
    - Garage
    - Hallway
    - Closet
- Variations of the rooms are possibile (e.g. kids bedroom, Study room, Walk-in Closet, Library, Master bedroom, Open Kitchen, Home Gym, Playroom, ...)
- If an object is shared between two rooms or is located at the boundary of two rooms, the room location should be labeled as "between dining room and kitchen" or similar. No hard rule for this, `follow the flow :)`


## Object Deletion Rules
- Delete the annotation if all four images are black.
- Delete the annotation if three images are black and one is noisy.
- Delete the annotation if all images are non-informative (i.e., no meaningful annotation can be extracted).

## Picture Label Rules
- Retain only really relevant pictures:
  - Relevant pictures are those that are clearly visible and unique, or visible and easily locatable based on nearby objects.
- If multiple pictures are next to each other:
  - If there are more than three pictures, delete at least one.
  - If two pictures are side by side:
    - If both are small, delete both.
    - If both are large, retain both.
- Each scene and floor should have a maximum of 4 pictures.

## Pillow Label Rules
- If multiple pillows are next to each other, leave only one.
- For each scene and floor, there should be no more than two pillows.

## Books Labeling Rules
- Each book should be associated with only one location. For instance, if there are multiple libraries, only one book can be annotated per library.
- A maximum of two books can be annotated per scene and floor.

## Borderline Objects
- Borderline objects are those where it is unclear whether they should be deleted or retained. 
- In such cases, you have two options:
    - Document them (i.e., add them to a list of "borderline objects") and discuss them after completing the annotation process.
    - Delete them immediately if deemed unnecessary.


## Description Examples
    Original Description:

    ```json
    {
        "object_category": "rack",
        "object_id": "rack_9",
        "floor_id": 0,
        "description": [
            "wooden shelf with a deer head on it. it is located near a chair, picture frame, 
            blinds, a record player, and a speaker.",
        ],
    },
    ```

    Becomes:

    ```json
    {
        "object_category": "rack",
        "object_id": "rack_9",
        "floor_id": 0,
        "description": [
            "a wooden shelf rack with a deer head ornament on it. It is located between a chair
            and a picture frame depicting an old man.",
        ],
    },
    ```

