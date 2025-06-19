import os
import json
import numpy as np
import skimage.draw
from mrcnn.utils import Dataset  
from mrcnn.config import Config  
    


class CustomDataset(Dataset):

    def load_custom(self, dataset_dir, subset,FileType_path,annotation_json_path):
        """Load a subset of the Horse-Man dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        
        # Add classes. We have only one class to add.
        # self.add_class("object", 1, "silt_fence_type_1")
        # self.add_class("object", 2, "silt_fence_type_2")
        # self.add_class("object", 3, "silt_fence_type_3")
        # self.add_class("object", 4, "silt_fence_type_4")



        with open(FileType_path, 'r') as f:
            data = json.load(f)

        # 2. Safely get the list of types from the key "silt_fence_types"
        #    If the key isn't found, default to an empty list
        types = data.get("types", [])
        classes_count = 0
        # 3. Dynamically add each type using a loop
        for idx, fence_type in enumerate(types, start=1):
            # Here, 'object' is your super-category (could be anything you like)
            self.add_class("object", idx, fence_type)
            classes_count +=1

        print("classes count is : ", classes_count)







        # self.add_class("object", 3, "xyz") #likewise

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        
        annotations1 = json.load(open(os.path.join(dataset_dir, annotation_json_path)))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes'] for s in a['regions']]
            print("objects:",objects)

            #name_dict = {"silt_fence_type_1": 1,"silt_fence_type_2": 2,"silt_fence_type_3": 3,"silt_fence_type_4": 4} #,"xyz": 3}
            name_dict = {type: idx for idx, type in enumerate(types, start=1)}
            # key = tuple(name_dict)
            num_ids = [name_dict[a['item_name']] for a in objects]

            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )
        return classes_count

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)


        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



def load_datasets(dataset_path,FileType_path,annotation_json_path):
    """
    A function to load and prepare both training and validation datasets.
    dataset_path: Path to the dataset directory.
    """
    # Training dataset
    dataset_train = CustomDataset()
    count = dataset_train.load_custom(dataset_path, "train",FileType_path,annotation_json_path)
    dataset_train.prepare()
    print(f"Training class names: {dataset_train.class_names}")

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(dataset_path, "train",FileType_path,annotation_json_path)  # Corrected "train" to "val"
    dataset_val.prepare()

    return dataset_train, dataset_val


class CustomConfig(Config):

    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 +6# Background + car and truck

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9



# class CustomConfig(Config):
#     """Configuration for training on the custom dataset.
#     Derives from the base Config class and overrides some values.
#     """

#     # Give the configuration a recognizable name
#     NAME = "object"

#     # We use a GPU with 12GB memory, which can fit two images.
#     # Adjust down if you use a smaller GPU.
#     IMAGES_PER_GPU = 1

#     def __init__(self, count=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)  # Call the parent constructor
#         self.NUM_CLASSES = 1 + count  # Background + number of classes in your dataset
#         self.STEPS_PER_EPOCH = 100  # Number of training steps per epoch
#         self.DETECTION_MIN_CONFIDENCE = 0.9  # Skip detections with < 90% confidence

#         print("-------------------------------------------.------------------------------------------------------------------count : ",self.NUM_CLASSES)