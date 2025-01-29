import torch
import torch.nn as nn
import numpy as np
import functools
from torch.optim import lr_scheduler
import torchvision.models as models
import torchvision.transforms as transforms
from models.swin_transformer import SwinTransformerSys
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou
import copy

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        with torch.no_grad():
          self.vgg_relu_3_3 = self.contentFunc(15)
          self.vgg_relu_2_2 = self.contentFunc(8)
          self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def contentFunc(self, relu_layer):
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == relu_layer:
                break

        return model

    def get_loss(self, fakeIm, realIm):

        fakeIm = self.transform(fakeIm)
        realIm = self.transform(realIm)

        f_fake_2_2 = self.vgg_relu_2_2.forward(fakeIm)
        f_real_2_2 = self.vgg_relu_2_2.forward(realIm)

        f_fake_3_3 = self.vgg_relu_3_3.forward(fakeIm)
        f_real_3_3 = self.vgg_relu_3_3.forward(realIm)

        f_real_2_2_no_grad = f_real_2_2.detach()
        f_real_3_3_no_grad = f_real_3_3.detach()
        mse = nn.MSELoss()
        loss = mse(f_fake_2_2, f_real_2_2_no_grad) + mse(f_fake_3_3, f_real_3_3_no_grad)
        return loss / 2.0

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)

class SwinTransformer_Backbone(nn.Module):
    def __init__(self, config, img_size=224, num_classes=3, zero_head=False, vis=False):
        super(SwinTransformer_Backbone, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x, x1, x2, x3):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        y, y1, y2, y3 = self.swin_unet(x, x1, x2, x3)
        return y, y1, y2, y3

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

        else:
            print("none pretrain")

class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
    
class TextBoxClassifier(nn.Module):

    def __init__(self):
        super(TextBoxClassifier, self).__init__()

    def forward(self, features):
        return 0
    

class CIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(CIoULoss, self).__init__()
        self.eps = eps

    def sort_bbox(self, bbox):
        return sorted(bbox, key=lambda x: x[0])

    def get_loss(self, pred_boxes, true_boxes, ciou_threshold=0.2):

        max_loss = 1
        if len(pred_boxes) == 0:
            return max_loss

        pred_boxes = torch.stack(pred_boxes)
        true_boxes = torch.stack(true_boxes)
        pred_boxes = pred_boxes.float().to(pred_boxes.device)
        true_boxes = true_boxes.float().to(true_boxes.device)

        total_ciou = []

        for true_box in true_boxes:
            ciou_scores = bbox_iou(true_box,pred_boxes, xywh=True, CIoU=True)
            best_ciou = ciou_scores.max()
            if best_ciou < ciou_threshold:
                best_ciou = 0
            total_ciou.append(best_ciou)

        total_ciou = np.array(total_ciou)
        ciou_loss =  (1-total_ciou).mean().item()

        return ciou_loss
    
    def __call__(self, pred_results, true_boxes):
        sorted_pred_boxes = self.sort_bbox(pred_results[0].boxes.xywhn)
        return self.get_loss(sorted_pred_boxes, true_boxes)


class CELoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(CELoss, self).__init__()
        self.eps = eps

    def select_indices_target_boxes(self, pred_boxes, true_boxes, ciou_threshold=0.3):

        matched_indices = {}
        if len(pred_boxes) == 0:
            return matched_indices

        pred_boxes = torch.stack(pred_boxes)
        true_boxes = torch.stack(true_boxes)
        pred_boxes = pred_boxes.float().to(pred_boxes.device)
        true_boxes = true_boxes.float().to(true_boxes.device)

        for idx, pred_box in enumerate(pred_boxes):
            ciou_scores = bbox_iou(pred_box,true_boxes, xywh=True, CIoU=True)
            best_ciou= ciou_scores.max()
            matched_idx = torch.argmax(ciou_scores)
            if best_ciou > ciou_threshold:
                matched_indices[idx] = matched_idx.item()

        return matched_indices
    
    def sort_bbox(self, bbox):
        return sorted(bbox, key=lambda x: x[0])
    
    def sort_class_log_prob(self,boxes, log_prob):
        x1_coordinates = boxes[:, 0]
        sorted_indices = torch.argsort(x1_coordinates)
        sorted_log_prob = log_prob[0][sorted_indices]
        sorted_log_prob = torch.log(sorted_log_prob.clamp(min=1e-12))
        return sorted_log_prob
    
    def find_corresponding_correct_label(self,matched_indices,sorted_labels):
        relevant_indices = list(matched_indices.values())
        relevant_values = sorted_labels[relevant_indices]
        return relevant_values
    
    def select_relevant_log_prob(self,sorted_class_log_prob,matched_indices):
        relevant_indices = list(matched_indices.keys())
        relevant_values = sorted_class_log_prob[relevant_indices]
        return relevant_values

    def get_loss(self, log_probs, true_cls, num_missed_boxes, epsilon, loss_cls_max=3.58):
        max_loss = 1
        if len(log_probs)<1 or len(true_cls)<1:
            return max_loss * len(true_cls)
        
        missed_box_loss = max_loss * num_missed_boxes

        ce_loss = nn.CrossEntropyLoss(reduction='mean')
        loss_cls = ce_loss(log_probs, true_cls)
        loss_cls = min((loss_cls.item() + missed_box_loss) / loss_cls_max, 1.0)
        return loss_cls
    
    def __call__(self, predicted_results, plate_info, epsilon=1e-9):
        max_loss = 1
        sorted_class_log_prob = self.sort_class_log_prob(predicted_results.boxes.xywh, predicted_results.class_logits)
        sorted_fake_B_PlateNum = self.sort_bbox(predicted_results.boxes.xywhn)
        matched_indices = self.select_indices_target_boxes(sorted_fake_B_PlateNum, plate_info["sorted_boxes_xywhn"])
        print("matched_indices",matched_indices)
        mapped_labels = self.find_corresponding_correct_label(matched_indices,plate_info["sorted_labels"][0])
        mapped_labels = mapped_labels.long()
        relevant_log_prob = self.select_relevant_log_prob(sorted_class_log_prob,matched_indices)
        print("mapped_labels",mapped_labels)
        if len(mapped_labels)<1:
            return max_loss
        num_missed_boxes = abs(len(plate_info["sorted_labels"][0]) - len(mapped_labels))
        return self.get_loss(relevant_log_prob, mapped_labels, num_missed_boxes, epsilon)
    

class PlateNumAccurate(nn.Module):
    def __init__(self, eps=1e-7):
        super(PlateNumAccurate, self).__init__()
        self.eps = eps

    def sort_bbox(self, bbox):
        return sorted(bbox, key=lambda x: x[0])
    
    def select_indices_target_boxes(self, pred_boxes, true_boxes, ciou_threshold=0.3):

        matched_indices = {}
        if len(pred_boxes) == 0:
            return matched_indices

        pred_boxes = torch.stack(pred_boxes)
        true_boxes = torch.stack(true_boxes)
        pred_boxes = pred_boxes.float().to(pred_boxes.device)
        true_boxes = true_boxes.float().to(true_boxes.device)

        for idx, pred_box in enumerate(pred_boxes):
            ciou_scores = bbox_iou(pred_box,true_boxes, xywh=True, CIoU=True)
            best_ciou= ciou_scores.max()
            matched_idx = torch.argmax(ciou_scores)
            if best_ciou > ciou_threshold:
                matched_indices[idx] = matched_idx.item()

        return matched_indices
    
    def find_corresponding_correct_label(self,matched_indices,sorted_labels):
        relevant_indices = list(matched_indices.values())
        relevant_values = sorted_labels[relevant_indices]
        return relevant_values


    def __call__(self, pred_boxes, plate_info):
        if len(pred_boxes.boxes.xywhn) != len(plate_info["sorted_boxes_xywhn"]):
            return 0
        
        sorted_fake_B_PlateNum = self.sort_bbox(pred_boxes.boxes.xywhn)
        matched_indices = self.select_indices_target_boxes(sorted_fake_B_PlateNum, plate_info["sorted_boxes_xywhn"])

        if len(matched_indices) != len(plate_info["sorted_labels"][0]):
            return 0
        
        mapped_labels = self.find_corresponding_correct_label(matched_indices,plate_info["sorted_labels"][0])
        mapped_labels = mapped_labels.long()

        if torch.equal(mapped_labels,plate_info["sorted_labels"][0]):
            return 1
        
        return 0

class FeatureExtractor(nn.Module):
    def __init__(self,model_path):
        super(FeatureExtractor, self).__init__()
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, fake_B):
        results= self.model.predict(fake_B,device = self.device, embed=[4,12,18])
        return results[0]

    