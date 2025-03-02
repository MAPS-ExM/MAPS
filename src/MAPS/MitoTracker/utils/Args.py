import torch
import yaml
import os
import warnings
from typing import Optional, List


class SelectedLoss:
    def __init__(self, loss_fn: str) -> None:
        self.loss_functions = loss_fn.lower().split(' ')

    def __getitem__(self, loss):
        return loss.lower() in self.loss_functions


class Args:
    """ "
    Reads in a yaml file <base_yaml_file> and saves the parameters in a dictionary.
    Optionally, an additional yaml file <extension_yaml_file> is read in and the values of the
    base dictionary are replaced with the new values for the parameters present in the new file.
    Additionally performs some preprocessing of some of the parameters like converting strings
    of scientific notation to python floats and checking that paths end with a trailing `/`.
    """

    def __init__(self, base_yaml_file: str, extension_yaml_file: Optional[str] = None):
        # Load the yaml file and update the corresponding attributes
        if isinstance(base_yaml_file, str) and os.path.isfile(base_yaml_file):
            with open(base_yaml_file, 'r') as f:
                config_dict = yaml.safe_load(f)
        # The file can also be given as a direct string and then is read in directly (useful for debugging)
        elif isinstance(base_yaml_file, dict):
            config_dict = yaml.safe_load(base_yaml_file)
        else:
            raise ValueError(f'{base_yaml_file} is neither a file nor a dict to load the config')
        self.__dict__.update(**config_dict)

        # If an extension yaml file is given, update the parameters according to this extension file:
        # This is useful if I run several experiments with one base configuration and want to see the
        # effects of isolated changes in the extension_yaml_file
        if extension_yaml_file is not None:
            if isinstance(extension_yaml_file, str) and os.path.isfile(extension_yaml_file):
                with open(extension_yaml_file, 'r') as f:
                    extension_dict = yaml.safe_load(f)
            else:
                extension_dict = yaml.safe_load(extension_yaml_file)

            for key, value in extension_dict.items():
                if key in vars(self):
                    setattr(self, key, value)
                else:
                    warnings.warn(f'Key {key} is in the extension yaml file but not in base configuration!')
                    setattr(self, key, value)

            # Save input args to save them later as yaml file for later use (like in predictions)
            # When running experiments this is helpful as the original args file might not be easily available
            config_dict.update(extension_dict)
        self.orig_args = config_dict

        # Transform some data
        if self.device_id != 'all':
            self.device = torch.device('cuda:' + str(self.device_id)) if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        for var in ['lr', 'l2_decay']:
            self.convert_scientific_notation_to_float(var)
        for var in ['debug_n_samples', 'input_channels']:
            self.convert_to_int(var)

        # Loss function
        if 'loss_fn' in vars(self):
            self.selected_loss = SelectedLoss(self.loss_fn)

        # Set some defaults
        if 'log_stdout' not in vars(self):
            self.log_stdout = True

        self._add_path_prefix('path_root', ['path_img', 'path_target'])
        self._add_path_prefix('path_root_img', ['path_img_train', 'path_img_val', 'path_img_test'])
        self._add_path_prefix('path_root_target', ['path_target_train', 'path_target_val', 'path_target_test'])
        self._add_path_prefix('path_root_mitomasks', ['path_masks_train', 'path_masks_val', 'path_masks_test'])
        self._add_path_prefix('path_root_boundary', ['path_boundary_train', 'path_boundary_val', 'path_boundary_test'])
        self._add_path_prefix('path_root', ['path_img_src', 'path_target_src'])
        self._add_path_prefix('path_trg', ['path_img_trg', 'path_target_trg'])
        self._add_path_prefix('path_src', ['path_img_src2', 'path_target_src2'])

        # Combine normal and drug data if present
        if getattr(self, 'path_img_train_drug', None) is not None:
            self._add_path_prefix(
                'path_root_img_drug', ['path_img_train_drug', 'path_img_val_drug', 'path_img_test_drug']
            )
            self._add_path_prefix(
                'path_root_target_drug', ['path_target_train_drug', 'path_target_val_drug', 'path_target_test_drug']
            )
            self.path_img_train += self.path_img_train_drug
            self.path_img_val += self.path_img_val_drug
            self.path_img_test += getattr(self, 'path_img_test_drug', [])
            self.path_target_train += self.path_target_train_drug
            self.path_target_val += self.path_target_val_drug
            self.path_target_test += getattr(self, 'path_target_test_drug', [])

        # Consistency checks
        if not self.run_name.lower().startswith('debug'):
            if 'path_img_train' in vars(self):
                # Check that training and testing are disjunct
                if len(set.intersection(set(self.path_img_train), set(self.path_img_test))) != 0:
                    print('WARNING: Training and testing sets are not disjunct!')

        # Legacy handling: Make some changes to keep some code after refactoring running
        if self.phase == 3:
            self.phase = 'inner_segmentation'

    def _add_path_prefix(self, prefix_attr: str, path_attr_list: List[str]):
        """Add <prefix_attr> to  every entry in <path_attr_list>"""
        for p in path_attr_list:
            try:
                appended_paths = [os.path.join(getattr(self, prefix_attr), cur_p) for cur_p in getattr(self, p)]
                setattr(self, p, appended_paths)
            except AttributeError:
                pass

    def to_string(self) -> str:
        """
        Builds a long string with every parameters on its own line. Used for documentation of experiments
        """
        doc = 'Configurations: \n'
        doc += ' \n'.join(k + ': ' + str(v) for k, v in self.__dict__.items())
        return doc

    def convert_scientific_notation_to_float(self, name: str) -> None:
        """
        Convert strings like '1e-5' into the corresponding float number
        """
        try:
            setattr(self, name, float(getattr(self, name)))
        except AttributeError as e:
            print(f'While converting args to float: ', e, sep='')

    def convert_to_int(self, name: str) -> None:
        try:
            setattr(self, name, int(getattr(self, name)))
        except AttributeError:
            pass
