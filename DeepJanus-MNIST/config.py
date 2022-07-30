DJ_DEBUG = 1

# GA Setup
POPSIZE = 100

STOP_CONDITION = "iter"
#STOP_CONDITION = "time"

NGEN = 15
RUNTIME = 3600
STEPSIZE = 10
# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6

# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 10

K_SD = 0.1

# K-nearest
K = 1

# Archive configuration
ARCHIVE_THRESHOLD = 4.0

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10


INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False

MODEL2 = 'models/cnnClassifier_lowLR.h5'
MODEL = 'models/cnnClassifier.h5'
#MODEL = "models/regular3"
#MODEL = 'models/cnnClassifier_001.h5'
#MODEL = 'models/cnnClassifier_op.h5'

RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'original_dataset/janus_dataset_comparison.h5'
EXPLABEL = 5

#TODO: set interpreter
INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'

#Attention Maps Options:

MUTANTS_ROOT_FOLDER = "mutants/debug/"    
METHOD_LIST = ["remut"]
# METHOD_LIST = ["NOremut"]
# ATTENTION_METHOD = "mth5"
# ATTENTION_METHOD = "mth1"
ATTENTION_METHOD = "distances"
SAVE_IMAGES = True
N = 54 #Nummber of Images to take from dataset
START_INDEX_DATASET = 0
EXTENT = 0.1
NUMBER_OF_POINTS = 2
SQUARE_SIZE = 2
NUMBER_OF_MUTATIONS = 1000
NUMBER_OF_REPETITIONS = 5
RANDOM_SEED = 1
SHUFFLE_IMAGES = False
NUMBER_OF_DIGIT_SAMPLES = 5 #Number of samples of a same digit that will be mutated in the experiment
SEEDS_LIST = [4398, 980, 987423, 99982, 1123, 4098, 1946, 22601, 55037, 812109, 53898, 187988]

SAVE_STATS4_CSV = False #Option to print the location of the points mutated at each iteration