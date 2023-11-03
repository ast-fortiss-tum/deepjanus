from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

from attention_maps import input_reshape_and_normalize_images
from predictor import Predictor


class AttentionManager:
    score = CategoricalScore(0)
    replace2linear = ReplaceToLinear()
    gradcam = GradcamPlusPlus(Predictor.model,
                              model_modifier=replace2linear,
                              clone=True)

    @staticmethod
    def compute_attention_maps(images):  # images should have the shape: (x, 28, 28) where x>=1
        X = input_reshape_and_normalize_images(images)
        attention_maps = AttentionManager.gradcam(AttentionManager.score,
                                                  X,
                                                  penultimate_layer=-1)
        return attention_maps
