import torch
import numpy as np
from torch import nn
from shapely.geometry import Polygon

class IOU(nn.Module):

    def __init__(self, weight=None, size_average=True, flag="training"):

        super(IOU, self).__init__()


    def forward(self, inputs_, targets, flag):

        if flag == "training":

            inputs = torch.zeros_like(targets)

            ## same h
            inputs[:, 0] = inputs_[:,0] * 500 + 500  # x: [-1, 1] --> [0, 1000]
            inputs[:, 1] = inputs_[:,1] * 500 + 500  # y: [-1, 1] --> [0, 1000]
            inputs[:, 2] = targets[:,2]              # same h
            inputs[:, 3] = inputs_[:,2] * 50 + 50    # w: [-1, 1] --> [0, 100]  
            inputs[:, 4] = torch.sin(inputs_[:,3] * np.pi + np.pi) # sin theta: [-1, 1] --> [0, 2pi]
            inputs[:, 5] = torch.cos(inputs_[:,3] * np.pi + np.pi) # cos theta: [-1, 1] --> [0, 2pi]


            ## learn h
            # inputs[:, 0] = inputs_[:,0] * 500 + 500  # x: [-1, 1] --> [0, 1000]
            # inputs[:, 1] = inputs_[:,1] * 500 + 500  # y: [-1, 1] --> [0, 1000]
            # inputs[:, 2] = inputs_[:,2] * 50 + 50    # h: [-1, 1] --> [0, 100]
            # inputs[:, 3] = inputs_[:,3] * 50 + 50    # w: [-1, 1] --> [0, 100]  
            # inputs[:, 4] = torch.sin(inputs_[:,4] * np.pi + np.pi) # sin theta: [-1, 1] --> [0, 2pi]
            # inputs[:, 5] = torch.cos(inputs_[:,4] * np.pi + np.pi) # cos theta: [-1, 1] --> [0, 2pi]

            ## l2 loss
            loss = torch.mean(torch.pow(torch.sum(torch.pow(inputs - targets, 2), 1), 0.5))

            batch_size = inputs.shape[0]
            iou_mean = 0
            iou_max = -np.inf

            for iBox in range(batch_size):
                input_box = self.getCorners(inputs[iBox, :])
                target_box = self.getCorners(targets[iBox, :])
                iou = self.intersection_over_union(input_box, target_box)
                iou_mean += iou
                iou_max = max(iou_max, iou)

            iou_mean /= batch_size

            return loss, iou_mean, iou_max

        if flag == "validation":

            inputs = torch.zeros([targets.shape[0], targets.shape[2]])

            ## same h
            inputs[:, 0] = inputs_[:,0] * 500 + 500  # x: [-1, 1] --> [0, 1000]
            inputs[:, 1] = inputs_[:,1] * 500 + 500  # y: [-1, 1] --> [0, 1000]
            inputs[:, 2] = targets[:,:,2][:,0]              # same h
            inputs[:, 3] = inputs_[:,2] * 50 + 50    # w: [-1, 1] --> [0, 100]  
            inputs[:, 4] = torch.sin(inputs_[:,3] * np.pi + np.pi) # sin theta: [-1, 1] --> [0, 2pi]
            inputs[:, 5] = torch.cos(inputs_[:,3] * np.pi + np.pi) # cos theta: [-1, 1] --> [0, 2pi]

            ## learn h
            # inputs[:, 0] = inputs_[:,0] * 500 + 500  # x: [-1, 1] --> [0, 1000]
            # inputs[:, 1] = inputs_[:,1] * 500 + 500  # y: [-1, 1] --> [0, 1000]
            # inputs[:, 2] = inputs_[:,2] * 50 + 50    # h: [-1, 1] --> [0, 100]
            # inputs[:, 3] = inputs_[:,3] * 50 + 50    # w: [-1, 1] --> [0, 100]  
            # inputs[:, 4] = torch.sin(inputs_[:,4] * np.pi + np.pi) # sin theta: [-1, 1] --> [0, 2pi]
            # inputs[:, 5] = torch.cos(inputs_[:,4] * np.pi + np.pi) # cos theta: [-1, 1] --> [0, 2pi]

            batch_size = inputs.shape[0]
            iou_max = -np.inf
            iou_max_sum = 0
            score = 0

            for iImage in range(batch_size):
                
                for iBox in range(targets.shape[1]):
                    input_box = self.getCorners(inputs[iImage, :])
                    target_box = self.getCorners(targets[iImage, iBox, :])
                    iou = self.intersection_over_union(input_box, target_box)

                    # iou_max = max(iou_max, iou)
                    if iou > iou_max:
                        iou_max = iou
                        index_iou_max = iBox

                theta_model = inputs_[index_iou_max, -1] * np.pi + np.pi   
                sin_theta_ref = targets[iImage, index_iou_max, :][-2] #TO CHECK!!!
                cos_theta_ref = targets[iImage, index_iou_max, :][-1] #TO CHECK!!!
                theta_ref = np.arctan2(cos_theta_ref, sin_theta_ref) #TO CHECK!!!
                score += 1 * ((iou_max > 0.25) and (torch.abs(theta_model-theta_ref) < (np.pi/6)))
                iou_max_sum += iou_max

            score /= (batch_size)
            iou_max_sum /= (batch_size)

            return score, iou_max_sum



    @staticmethod
    def intersection_over_union(box_1, box_2):
        box1 = Polygon([box_1[0], box_1[1], box_1[2], box_1[3]])
        box2 = Polygon([box_2[0], box_2[1], box_2[2], box_2[3]])
        iou = box1.intersection(box2).area / box1.union(box2).area
        return iou

    @staticmethod
    def l2_distance(input, target):
        x_input, y_input, _, _, _, _ = input
        x_target, y_target, _, _, _, _ = target 
        l2_dist = ((x_input - x_target)**2 + (y_input - y_target)**2)**0.5
        return l2_dist


    @staticmethod
    def getCorners(output):

        x, y, h, w, sin_, cos_ = output
        
        rot_matrix = np.array([ [cos_,-sin_], [sin_, cos_] ])
        
        xmin = -w/2
        xmax = w/2
        ymin = -h/2
        ymax = h/2

        P1 = np.array([xmin, ymin])
        P2 = np.array([xmax, ymin])
        P3 = np.array([xmax, ymax])
        P4 = np.array([xmin, ymax])
        
        P1 = np.matmul(rot_matrix, P1)
        P2 = np.matmul(rot_matrix, P2)
        P3 = np.matmul(rot_matrix, P3)
        P4 = np.matmul(rot_matrix, P4)
        
        P1 += np.array([x, y])
        P2 += np.array([x, y])
        P3 += np.array([x, y])
        P4 += np.array([x, y])

        return (P1, P2, P3, P4)


