import sys
sys.path.insert(0, './yolact_vizta')

import argparse
import os
import platform
import shutil
import time
import re
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from yolact_vizta.modules.yolact import Yolact
from yolact_vizta.config import get_config
from yolact_vizta.utils.coco import COCODetection, detect_collate
from yolact_vizta.utils import timer
from yolact_vizta.utils.output_utils import nms, after_nms, draw_img
from yolact_vizta.utils.common_utils import ProgressBar
from yolact_vizta.utils.augmentations import val_aug

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.utils.plots import Annotator, colors

def detect(args):
    deep_sort_weights, save_txt, out= args.deep_sort_weights, args.save_txt, args.output

    if args.image is not None:
        source = args.image
    else:
        source = args.video

    
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    prefix = re.findall(r'best_\d+\.\d+_', args.weight)[0]
    suffix = re.findall(r'_\d+\.pth', args.weight)[0]
    args.cfg = args.weight.split(prefix)[-1].split(suffix)[0]
    cfg = get_config(args, mode='detect')

    net = Yolact(cfg)
    net.load_weights(cfg.weight, cfg.cuda)
    net.eval()
    print(f'Model loaded with {cfg.weight}.\n')

    if cfg.cuda:
        cudnn.benchmark = True
        cudnn.fastest = True
        net = net.cuda()
    
    save_path = str(Path(out))
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    if cfg.image is not None:
        dataset = COCODetection(cfg, mode='detect')
        data_loader = data.DataLoader(dataset, 1, num_workers=4, shuffle=False,
                                        pin_memory=True, collate_fn=detect_collate)
        ds = len(data_loader)
        leading_zeros = len(str(ds)) + 1
        assert ds > 0, 'No .png images found.'
        progress_bar = ProgressBar(40, ds)
        timer.reset()

        for i, (img, img_origin, img_name) in enumerate(data_loader):
            if i == 1:
                timer.start()

            if cfg.cuda:
                img = img.cuda()

            img_h, img_w = img_origin.shape[0:2]

            with torch.no_grad(), timer.counter('forward'):
                class_p, box_p, coef_p, proto_p = net(img)

            with timer.counter('nms'):
                ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, net.anchors, cfg)

            with timer.counter('after_nms'):
                ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg, img_name=img_name)
                # addition for deepsort
                for k, det in enumerate(boxes_p):
                    annotator = Annotator(img, line_width=2, pil=not ascii)
                    if det is not None and len(det):

                        xywhs = det[0:4] # contains x-top left, y-top left, width, height
                        xywhs[0] += (xywhs[2]/2) # from x-top left to x-center
                        xywhs[1] += (xywhs[3]/2) # from y-top left to y-center
                        confs = class_p[k]
                        clss = 1

                        # pass detections to deepsort
                        if cfg.cuda:
                            outputs = deepsort.update(xywhs.cuda(), confs.cuda(), clss.cuda(), img)
                        else:
                            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img)
                        
                        # draw boxes for visualization
                        if len(outputs) > 0:
                            for j, (output, conf) in enumerate(zip(outputs, confs)): 
                                
                                bboxes = output[0:4]
                                id = output[4]
                                cls = output[5]
                                names = 'person'
                                c = int(cls)  # integer class
                                label = f'{id} {names} {conf:.2f}'
                                annotator.box_label(bboxes, label, color=colors(c, True))

                                if save_txt:
                                    # to MOT format
                                    bbox_left = output[0]
                                    bbox_top = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]
                                    # Write MOT compliant results to file
                                    with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (i + 1, id, bbox_left,
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                    
                    else:
                        deepsort.increment_ages()
           
            with timer.counter('save_img'):
                img_name = str(i).zfill(leading_zeros) + ".png"
                img_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, img_origin, cfg, img_name=img_name)
                cv2.imwrite(save_path + f'/{img_name}', img_numpy)

            aa = time.perf_counter()
            if i > 0:
                batch_time = aa - temp
                timer.add_batch_time(batch_time)
            temp = aa

            if i > 0:
                t_t, t_d, t_f, t_nms, t_an, t_si = timer.get_times(['batch', 'data', 'forward',
                                                                    'nms', 'after_nms', 'save_img'])
                fps, t_fps = 1 / (t_d + t_f + t_nms + t_an), 1 / t_t
                bar_str = progress_bar.get_bar(i + 1)
                print(f'\rTesting: {bar_str} {i + 1}/{ds}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                        f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                        f't_after_nms: {t_an:.3f} | t_save_img: {t_si:.3f}', end='')

        print('\nFinished, saved in: inference/output.')

    # detect videos
    elif cfg.video is not None:
        vid = cv2.VideoCapture(cfg.video)

        target_fps = round(vid.get(cv2.CAP_PROP_FPS))
        frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        name = cfg.video.split('/')[-1]
        video_writer = cv2.VideoWriter(f'results/videos/{name}', cv2.VideoWriter_fourcc(*"mp4v"), target_fps,
                                        (frame_width, frame_height))

        progress_bar = ProgressBar(40, num_frames)
        timer.reset()
        t_fps = 0

        for i in range(num_frames):
            if i == 1:
                timer.start()

            frame_origin = vid.read()[1]
            img_h, img_w = frame_origin.shape[0:2]
            frame_trans = val_aug(frame_origin, cfg.img_size)

            frame_tensor = torch.tensor(frame_trans).float()
            if cfg.cuda:
                frame_tensor = frame_tensor.cuda()

            with torch.no_grad(), timer.counter('forward'):
                class_p, box_p, coef_p, proto_p = net(frame_tensor.unsqueeze(0))

            with timer.counter('nms'):
                ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, net.anchors, cfg)

            with timer.counter('after_nms'):
                ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg)
                # addition for deepsort
                for k, det in enumerate(boxes_p):
                    annotator = Annotator(img, line_width=2, pil=not ascii)
                    if det is not None and len(det):

                        xywhs = det[0:4] # contains x-top left, y-top left, width, height
                        xywhs[0] += (xywhs[2]/2) # from x-top left to x-center
                        xywhs[1] += (xywhs[3]/2) # from y-top left to y-center
                        confs = class_p[k]
                        clss = 1

                        # pass detections to deepsort
                        if cfg.cuda:
                            outputs = deepsort.update(xywhs.cuda(), confs.cuda(), clss.cuda(), img)
                        else:
                            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img)
                        
                        # draw boxes for visualization
                        if len(outputs) > 0:
                            for j, (output, conf) in enumerate(zip(outputs, confs)): 
                                
                                bboxes = output[0:4]
                                id = output[4]
                                cls = output[5]
                                names = 'person'
                                c = int(cls)  # integer class
                                label = f'{id} {names} {conf:.2f}'
                                annotator.box_label(bboxes, label, color=colors(c, True))

                                if save_txt:
                                    # to MOT format
                                    bbox_left = output[0]
                                    bbox_top = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]
                                    # Write MOT compliant results to file
                                    with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (i + 1, id, bbox_left,
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                    
                    else:
                        deepsort.increment_ages()

            with timer.counter('save_img'):
                frame_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, frame_origin, cfg, fps=t_fps)

            if cfg.real_time:
                cv2.imshow('Detection', frame_numpy)
                cv2.waitKey(1)
            else:
                video_writer.write(frame_numpy)

            aa = time.perf_counter()
            if i > 0:
                batch_time = aa - temp
                timer.add_batch_time(batch_time)
            temp = aa

            if i > 0:
                t_t, t_d, t_f, t_nms, t_an, t_si = timer.get_times(['batch', 'data', 'forward',
                                                                    'nms', 'after_nms', 'save_img'])
                fps, t_fps = 1 / (t_d + t_f + t_nms + t_an), 1 / t_t
                bar_str = progress_bar.get_bar(i + 1)
                print(f'\rDetecting: {bar_str} {i + 1}/{num_frames}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                        f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                        f't_after_nms: {t_an:.3f} | t_save_img: {t_si:.3f}', end='')

        if not cfg.real_time:
            print(f'\n\nFinished, saved in: results/videos/{name}')

        vid.release()
        video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument('--weight', default=None, type=str)
    parser.add_argument('--image', default=None, type=str, help='The folder of images for detecting.')
    parser.add_argument('--video', default=None, type=str, help='The path of the video to evaluate.')
    parser.add_argument('--img_size', type=int, default=512, help='The image size for validation.')
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
    parser.add_argument('--hide_mask', default=False, action='store_true', help='Hide masks in results.')
    parser.add_argument('--hide_bbox', default=False, action='store_true', help='Hide boxes in results.')
    parser.add_argument('--hide_score', default=False, action='store_true', help='Hide scores in results.')
    parser.add_argument('--cutout', default=False, action='store_true', help='Cut out each object and save.')
    parser.add_argument('--save_lincomb', default=False, action='store_true', help='Show the generating process of masks.')
    parser.add_argument('--no_crop', default=False, action='store_true',
                        help='Do not crop the output masks with the predicted bounding box.')
    parser.add_argument('--real_time', default=False, action='store_true', help='Show the detection results real-timely.')
    parser.add_argument('--visual_thre', default=0.5, type=float,
                        help='Detections with a score under this threshold will be removed.')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    args = parser.parse_args()
    

    with torch.no_grad():
        detect(args)