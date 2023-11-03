import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import join, exists
from os import makedirs

from folder import Folder
import rasterization_tools
from digit_mutator import DigitMutator


class Individual:
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    #COUNT = 0

    def __init__(self, id, desc, label):
        self.id = id
        self.expected_label = label
        self.xml_desc = desc
        self.purified = rasterization_tools.rasterize_in_memory(self.xml_desc)
        self.predicted_label = None
        self.confidence = None
        self.misclass = None
        self.attention = None
        self.mutate_attempts = None
        #Individual.COUNT += 1

    def reset(self):
        self.predicted_label = None
        self.confidence = None
        self.misclass = None
        self.attention = None

    def to_dict(self, ind_id):
        return {'id': str(self.id),
                'sol_id': str(ind_id),
                'expected_label': str(self.expected_label),
                'predicted_label': str(self.predicted_label),
                'misbehaviour': str(not self.misclass),
                'confidence': str(self.confidence),
                #'timestamp': str(self.timestamp),
                #'elapsed': str(self.elapsed_time),
        }

    def save_png(self, filename):
        plt.imsave(filename + '.png', self.purified.reshape(28, 28), cmap='gray', format='png')

    def dump(self, filename, ind_id):
        data = self.to_dict(ind_id)
        filedest = filename+".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def save_npy(self, filename):
        np.save(filename, self.purified)
        test_img = np.load(filename + '.npy')
        diff = self.purified - test_img
        assert (np.linalg.norm(diff) == 0)

    def save_svg(self, filename):
        data = self.xml_desc
        filedest = filename + ".svg"
        with open(filedest, 'w') as f:
            f.write(data)

    def export(self, ind_id):
        if not exists(Folder.DST_ARC):
            makedirs(Folder.DST_ARC)
        dst = join(Folder.DST_ARC, "mbr" + str(self.id))
        self.dump(dst, ind_id)
        self.save_npy(dst)
        self.save_png(dst)
        self.save_svg(dst)





