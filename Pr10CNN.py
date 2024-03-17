import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Load the pre-trained model and label map
model_path = 'path/to/your/model'
label_map_path = 'path/to/your/label_map.pbtxt'

detection_graph = tf.saved_model.load(model_path)
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

def load_image_into_numpy_array(image_path):
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Object detection function
def detect_objects(image_path):
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detection_graph(input_tensor)

    # Visualization of the results
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np[0],
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8
    )

    return image_np[0]

# Example usage
image_path = 'path/to/your/image.jpg'
output_image = detect_objects(image_path)
plt.imshow(output_image)
plt.show()
