import torch
import torch.nn as nn
from torchvision import models
from classification.simplenet import SimpleNet
from classification.basicnet import BasicNet


class NetLoader:

    def __init__(self, model_name, num_classes, resize_size=224,
                 feature_extract=True, use_pretrained=True, custom_pre_model=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.resize_size = resize_size
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        self.custom_pre_model = custom_pre_model

        self.model, self.input_size = self.initialize_model()

    def get_model(self):
        return self.model, self.input_size

    def initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if self.model_name == "basicnet":
            """ BasicNet
            """
            model_ft = BasicNet(classes=self.num_classes)
            input_size = self.resize_size
            # TODO check for None
            if self.use_pretrained and self.custom_pre_model:
                checkpoint = torch.load(self.custom_pre_model)
                model_ft.load_state_dict(checkpoint)
                model_ft.train()

        elif self.model_name == "simplenet":
            """ SimpleNet
            """
            model_ft = SimpleNet(classes=self.num_classes)
            input_size = self.resize_size
            if self.use_pretrained and self.custom_pre_model:
                checkpoint = torch.load(self.custom_pre_model)
                model_ft.load_state_dict(checkpoint)
                model_ft.train()

        elif self.model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "resnet34":
            """ Resnet34
            """
            model_ft = models.resnet34(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "resnet101":
            """ Resnet101
            """
            model_ft = models.resnet101(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "resnet152":
            """ Resnet152
            """
            model_ft = models.resnet152(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = self.num_classes
            input_size = 224

        elif self.model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif  self.model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained= self.use_pretrained, aux_logits=False)
            self.set_parameter_requires_grad(model_ft,  self.feature_extract)
            # Handle the auxilary net
            # num_ftrs = model_ft.AuxLogits.fc.in_features
            # model_ft.AuxLogits.fc = nn.Linear(num_ftrs,  self.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
