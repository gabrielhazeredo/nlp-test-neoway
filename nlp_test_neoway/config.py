from os import path
import os

import nlp_test_neoway

base_path = path.dirname(path.dirname(nlp_test_neoway.__file__))
workspace_path = path.join(base_path, 'workspace')
data_path = path.join(workspace_path, 'data')
models_path = path.join(workspace_path, 'models')
