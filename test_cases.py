from models import *

class Test_case(object):
    def __init__(self, test_case_num):
        """Initialization."""
        super(Test_case, self).__init__()
        if test_case_num == 1:
            self.net = ResNet_reduced_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 2:
            self.net = ResNet_reduced_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 3:
            self.net = ResNet_reduced_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'None'

        elif test_case_num == 4:
            self.net = ResNet_reduced_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.1
            self.use_data_augmentation = 'None'

        elif test_case_num == 5:
            self.net = ResNet_reduced_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
            
        elif test_case_num == 6:
            self.net = ResNet_reduced_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        else:
            raise ValueError('No matching test cases')