# Import needed packages
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.autograd import Variable
import os
import time
from classification.netloader import NetLoader


class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg
        #torch.manuel_seed(self.cfg.classificator.seed)
        #torch.backends.cudnn.deterministic = True
        #torch.bakcends.cudnn.benchmark = False
        print(f"[Classificator] Trainer Seed: {self.cfg.classificator.seed}")
        self.net = self.cfg.classificator.net
        self.classes = self.cfg.classificator.classes
        self.dataset_dir = self.cfg.classificator.dataset_path
        self.model_dir = self.cfg.classificator.model_path
        self.model_name = self.cfg.classificator.model_name
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.custom_pre_model = self.cfg.classificator.custom_pre_model
        self.batch_size = self.cfg.classificator.batch_size
        self.resize_size = self.cfg.classificator.resize_size
        self.cuda_avail = torch.cuda.is_available()
        # Flag for feature extracting. When False, we fine-tune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = self.cfg.classificator.feature_extract
        self.use_pretrained = self.cfg.classificator.use_pretrained

        net_loader = NetLoader(model_name=self.net, num_classes=self.classes,
                               resize_size=self.resize_size, feature_extract=self.feature_extract,
                               use_pretrained=self.use_pretrained, custom_pre_model=self.custom_pre_model)
        self.model, self.resize_size = net_loader.get_model()

        # Define transformations for the sets
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Load the training set
        train_set = datasets.ImageFolder(os.path.join(self.dataset_dir, "train"), self.data_transforms["train"])
        self.num_train = len(train_set)
        print("Train images: %s" % self.num_train)

        # Create a loader for the training set
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

        # Load the test set, note that train is set to False
        val_set = datasets.ImageFolder(os.path.join(self.dataset_dir, "val"), self.data_transforms["val"])
        self.num_test = len(val_set)
        print("Validation images: %s" % self.num_test)

        # Create a loader for the val set, note that both shuffle is set to false for the val loader
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # Create model, optimizer and loss function

        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()

        # Tensorboard
        self.writer = SummaryWriter(f"tensorboard_trainer_{self.model_name}")
        images, labels = next(iter(self.train_loader))
        img_grid = make_grid(images)
        self.writer.add_image(f"{self.cfg.classificator.model_name}_images", img_grid)
        self.writer.add_graph(self.model, images)

        if self.cuda_avail:
            self.model.cuda()

    # Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
    def adjust_learning_rate(self, epoch):
        lr = 0.001

        if epoch > 180:
            lr = lr / 1000000
        elif epoch > 150:
            lr = lr / 100000
        elif epoch > 120:
            lr = lr / 10000
        elif epoch > 90:
            lr = lr / 1000
        elif epoch > 60:
            lr = lr / 100
        elif epoch > 30:
            lr = lr / 10

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save_models(self, epoch, time=0.0):
        torch.save(self.model.state_dict(), self.model_path + "_{}.model".format(epoch))
        print("Checkpoint saved (time: %s min)" % time)

    def test(self):
        self.model.eval()
        test_acc = 0.0
        for i, (images, labels) in enumerate(self.val_loader):

            if self.cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Predict classes using images from the test set
            outputs = self.model(images)
            _, prediction = torch.max(outputs.data, 1)
            prediction = prediction
            test_acc += torch.sum(prediction == labels.data)

        # Compute the average acc and loss over all 10000 test images
        test_acc = test_acc / self.num_test

        return test_acc

    def train(self):
        best_acc = 0.0
        start = time.time()
        started = False

        num_epochs = self.cfg.classificator.epochs
        for epoch in range(num_epochs):
            self.model.train()
            train_acc = 0.0
            train_loss = 0.0
            for i, (images, labels) in enumerate(self.train_loader):
                # Move images and labels to gpu if available
                if self.cuda_avail:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                if not started:
                    print("Training started ...")
                    started = True

                # Clear all accumulated gradients
                self.optimizer.zero_grad()
                # Predict classes using images from the test set
                outputs = self.model(images)
                # Compute the loss based on the predictions and actual labels
                loss = self.loss_fn(outputs, labels)
                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                self.optimizer.step()

                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                train_acc += torch.sum(prediction == labels.data)

            # Call the learning rate adjustment function
            self.adjust_learning_rate(epoch)

            # Compute the average acc and loss over all 50000 training images
            train_acc = train_acc / self.num_train
            train_loss = train_loss / self.num_train

            # Evaluate on the test set
            test_acc = self.test()
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/validation", test_acc, epoch)
            self.writer.add_scalar("Loss/train", train_loss, epoch)

            # Print the metrics
            print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,
                                                                                            test_acc))

            # Save the model if the test acc is greater than our current best
            process_time = round((time.time() - start) / 60.0, 2)
            if test_acc > best_acc or epoch == num_epochs - 1:
                self.save_models(epoch, process_time)
                best_acc = test_acc

        # self.writer.close()

