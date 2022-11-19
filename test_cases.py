from models import *

class Test_case(object):
    def __init__(self, test_case_num):
        """Initialization."""
        super(Test_case, self).__init__()
        print("Test case num: ", test_case_num)
        if test_case_num == 1:
            self.net = ResNet_three_layer_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 2:
            self.net = ResNet_three_layer_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.001
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 3:
            self.net = ResNet_three_layer_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'None'

        elif test_case_num == 4:
            self.net = ResNet_three_layer_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.001
            self.use_data_augmentation = 'None'

        elif test_case_num == 5:
            self.net = ResNet_three_layer_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
            
        elif test_case_num == 6:
            self.net = ResNet_three_layer_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.001
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 7:
            self.net = ResNet_four_layer_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 8:
            self.net = ResNet_four_layer_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.001
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 9:
            self.net = ResNet_four_layer_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'None'

        elif test_case_num == 10:
            self.net = ResNet_four_layer_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.001
            self.use_data_augmentation = 'None'

        elif test_case_num == 11:
            self.net = ResNet_four_layer_1()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
            
        elif test_case_num == 12:
            self.net = ResNet_four_layer_1()
            self.optimizier_type = 'Adam'
            self.lr = 0.001
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 13:
            self.net = ResNet_three_layer_2()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'
        
        elif test_case_num == 14:
            self.net = ResNet_three_layer_3()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'
        
        elif test_case_num == 15:
            self.net = ResNet_three_layer_4()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'
        
        elif test_case_num == 16:
            self.net = ResNet_three_layer_5()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'
        
        elif test_case_num == 17:
            self.net = ResNet_three_layer_6()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 18:
            self.net = ResNet_three_layer_7()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 19:
            self.net = ResNet_three_layer_8()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'

        elif test_case_num == 20:
            self.net = ResNet_four_layer_2()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Naive'
        
        elif test_case_num == 21:
            self.net = ResNet_four_layer_2()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'

        elif test_case_num == 22:
            self.net = ResNet_three_layer_7()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'

        elif test_case_num == 23:
            self.net = ResNet_three_layer_8()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 24:
            self.net = ResNet_three_layer_9()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 25:
            self.net = ResNet_three_layer_10()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 26:
            self.net = ResNet_three_layer_11()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 27:
            self.net = ResNet_three_layer_12()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'

        elif test_case_num == 28:
            self.net = ResNet_four_layer_3()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 29:
            self.net = ResNet_four_layer_4()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 30:
            self.net = ResNet_four_layer_5()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 31:
            self.net = ResNet_four_layer_6()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        
        elif test_case_num == 32:
            self.net = ResNet_four_layer_7()
            self.optimizier_type = 'SGD'
            self.lr = 0.1
            self.use_data_augmentation = 'Auto'
        else:
            raise ValueError('No matching test cases')
