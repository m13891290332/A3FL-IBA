import sys
sys.path.append("../")
import time
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from models.resnet import ResNet18


# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Attacker:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.setup()
        self.trigger_model = Autoencoder().cuda()  # Initialize Autoencoder model
        self.trigger_optimizer = torch.optim.Adam(self.trigger_model.parameters(), lr=0.001)

    def setup(self):
        self.handcraft_rnds = 0

        # Initialize trigger
        self.trigger = torch.zeros((1, 3, 32, 32), requires_grad=False, device='cuda')  # Trigger pattern
        self.trigger[:, :, 2:2 + self.helper.config.trigger_size, 2:2 + self.helper.config.trigger_size] = 0.5

        # Initialize mask
        self.mask = torch.zeros((1, 3, 32, 32), requires_grad=False, device='cuda')  # Binary mask
        self.mask[:, :, 2:2 + self.helper.config.trigger_size, 2:2 + self.helper.config.trigger_size] = 1
        self.mask = self.mask.cuda()

    def init_badnets_trigger(self):
        print('Setup baseline trigger pattern.')
        self.trigger = torch.ones((1, 3, 32, 32), requires_grad=False, device='cuda') * 0.5
        self.trigger[:, 0, :, :] = 1

    def poison_input(self, inputs, labels, eval=False):
        """
        Poison a subset of the inputs using the dynamically updated trigger model.
        """
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])

        # Clone inputs to avoid inplace modification
        poisoned_inputs = self.trigger_model(inputs[:bkd_num])
        poisoned_inputs = poisoned_inputs * self.mask + inputs[:bkd_num] * (1 - self.mask)

        # Assign to a new Tensor to prevent inplace modification
        inputs = inputs.clone()
        inputs[:bkd_num] = poisoned_inputs

        labels = labels.clone()
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels

    def search_trigger(self, model, dl, type_, adversary_id=0, epoch=0):
        """
        Search for an optimized trigger pattern.
        """
        trigger_optim_time_start = time.time()
        K = self.helper.config.trigger_outter_epochs
        model.eval()

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t * m + (1 - m) * inputs
                    labels[:] = self.helper.config.target_class
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct / num_data
            return asr, total_loss

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config.trigger_lr
        t = self.trigger.clone()
        m = self.mask.clone()

        def grad_norm(gradients):
            """
            Calculate the gradient norm.
            """
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()

        for iter in range(K):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
                print(f"Iteration {iter}, ASR: {asr:.4f}, Loss: {loss:.4f}")

            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()

                # Clone to avoid inplace modification
                inputs = inputs.clone()
                poisoned_inputs = self.trigger_model(inputs)
                poisoned_inputs = poisoned_inputs * m + inputs * (1 - m)
                labels[:] = self.helper.config.target_class

                outputs = model(poisoned_inputs)
                loss = ce_loss(outputs, labels)

                # Optimize trigger model
                self.trigger_optimizer.zero_grad()
                loss.backward()
                self.trigger_optimizer.step()

                # Update static trigger (avoiding inplace)
                with torch.no_grad():
                    t = self.trigger_model(inputs[:1]).detach()
                    t = t * m + inputs[:1] * (1 - m)

        self.trigger = t.detach()  # Save the updated trigger
        trigger_optim_time_end = time.time()
        print(f"Trigger optimization completed in {trigger_optim_time_end - trigger_optim_time_start:.2f}s.")

    def adversarial_training(self, model, train_loader, optimizer, criterion, epsilon, alpha, pgd_steps):
        """
        Perform adversarial training with PGD attack.
        """
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            data.requires_grad = True

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass for adversarial example
            loss.backward()
            perturbed_data = data + alpha * data.grad.sign()
            perturbed_data = torch.clamp(perturbed_data, data - epsilon, data + epsilon)
            perturbed_data = torch.clamp(perturbed_data, 0, 1)

            # Zero gradients and perform adversarial training
            optimizer.zero_grad()
            output = model(perturbed_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        return model
