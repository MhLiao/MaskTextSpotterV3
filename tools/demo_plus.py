'''
Original code: demo.py, inference on a single sample.
Author: https://github.com/MhLiao

-----------

Adaption and modified for multiple sample inferece and TTA over Polygon on inferece time. 
Author: https://github.com/innat

Added: 
 - Inferece on multiple image files (within a directory)
 - Automate the output saving info into three following ways: .txt, .json, and visual prediction 
 - Added Test-Time-Augmentation over Polygons, applying Non-Max-Supression method
'''


from PIL import Image
from tta_utils import*
import tta_augment as TTA
import albumentations as A
from torchvision import transforms as T
import os, cv2, gc, json, argparse, numpy as np, warnings, torch 

from maskrcnn_benchmark.config import cfg
from albumentations.pytorch.transforms import ToTensorV2
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.chars import getstr_grid, get_tight_rect

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class TextDemo(object):
    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        min_image_size=224,
        output_polygon=True
    ):
        self.cfg   = cfg.clone()
        self.model = build_detection_model(cfg)
        
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.cpu_device           = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.output_polygon       = output_polygon

    def build_transform(self, step):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        
        if step == 1:
            print('Transformation Applying At Step: ', step)

            transform_one = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize(self.min_image_size),
                    T.ToTensor(),
                    to_bgr_transform,
                    normalize_transform,
                ]
            )
            return transform_one
        
        elif step == 2:
            print('Transformation Applying At Step: ', step)
            
            transform_two = T.Compose(
                [
                    T.ToPILImage(),
                    TTA.RandomContrast(prob=1),
                    TTA.RandomGamma(prob=1),
                    T.ToTensor(),
                    to_bgr_transform,
                    normalize_transform,
                ]
            )
            return transform_two
        
        elif step == 3:
            print('Transformation Applying At Step: ', step)
            
            transform_thr = T.Compose(
                [
                    T.ToPILImage(),
                    TTA.RandomContrast(prob=1),
                    TTA.RandomHue(prob=1),
                    T.ToTensor(),
                    to_bgr_transform,
                    normalize_transform,
                ]
            )
            return transform_thr
        
        else:
            print('Transformation Applying At Step: ', step)
            
            transform_thr = T.Compose(
                [
                    T.ToPILImage(),
                    TTA.RandomContrast(prob=1),
                    TTA.RandomSaturation(prob=1),
                    TTA.RandomBrightness(prob=1),
                    TTA.RandomHue(prob=1),
                    T.ToTensor(),
                    to_bgr_transform,
                    normalize_transform,
                ]
            )
            return transform_thr
        

    def compute_prediction(self, original_image_path, tta_step): # original_image, step
        for each_img_file in os.listdir(original_image_path):
            '''
            Iterate through all image files, which located in the original_image_path directory
            '''
            # a placeholder, that will contain polygons, transcription, score in a JSON format
            preds = [] 
            print()
            print('Loading File: ', each_img_file)
          
            # Test Time Augmentation
            for step in range(tta_step):
                print("#"*25)
                print("Start TTA Step ", str(step+1))
                
                # load the image 
                image = cv2.imread(os.path.join(original_image_path, each_img_file))
                if image is None: continue 
                temp_image = image.copy()
                    
                # adjust contrast and brightness 
                alpha = 1.5 # Contrast control (1.0-3.0)
                beta  = 0  # Brightness control (0-100)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
                # apply transformation to image, 
                # the transformation affect will be applied based on step number
                self.transforms   = self.build_transform(step+1)
                transformed_image = self.transforms(image) 

                # convert to an ImageList, padded so that it is divisible by
                # cfg.DATALOADER.SIZE_DIVISIBILITY
                image_list = to_image_list(transformed_image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
                image_list = image_list.to(self.device)
                
                # compute predictions
                with torch.no_grad():
                    predictions, _, _ = self.model(image_list)

                global_predictions = predictions[0]
                char_predictions   = predictions[1]

                char_mask  = char_predictions['char_mask']
                char_boxes = char_predictions['boxes']
                words, rec_scores = self.process_char_mask(char_mask, char_boxes)

                seq_words  = char_predictions['seq_outputs']
                seq_scores = char_predictions['seq_scores']
                global_predictions = [o.to(self.cpu_device) for o in global_predictions]

                # always single image is passed at a time
                global_prediction = global_predictions[0]

                # reshape prediction (a BoxList) into the original image size
                height, width     = image.shape[:-1] 
                global_prediction = global_prediction.resize((width, height))

                boxes  = global_prediction.bbox.tolist()
                scores = global_prediction.get_field("scores").tolist()
                masks  = global_prediction.get_field("mask").cpu().numpy()

                # a placeholder, it will contain result form each samples
                # above preds = [] is similar functionality, or preds = following three placholder
                # but I've sume different things to do, so saving the output again in this following placeholder
                result_polygons = []
                result_words    = []
                each_poly_score = []

                # iterate through the boxes, official 
                for k, box in enumerate(boxes):
                    score = scores[k]
                    if score < self.confidence_threshold: continue
                        
                    box     = list(map(int, box))
                    mask    = masks[k,0,:,:]
                    polygon = self.mask2polygon(mask, box, image.shape, 
                                                threshold=0.5, output_polygon=self.output_polygon)
                    
                    if polygon is None:
                        polygon = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]

                    result_polygons.append(polygon)
                    word            = words[k]
                    rec_score       = rec_scores[k]
                    seq_word        = seq_words[k]
                    seq_char_scores = seq_scores[k]
                    seq_score       = sum(seq_char_scores) / float(len(seq_char_scores))

                    if seq_score > rec_score:
                        result_words.append(seq_word)
                        each_poly_score.append(seq_score)
                    else:
                        result_words.append(word)
                        each_poly_score.append(rec_score)
                        
                                
                # we will create specific directory
                '''
                args.text_output <--- out_text
                --- <image_1>    <--- will create
                --- -- 1.text
                --- -- 2.text 
                --- -- 3.text    <--- output prediction from TTA, (1,2,3...are the step number of TTA)
                
                --- <image_2>    <--- will create
                <same as before>
                <same as before>
                .
                .
                so on!
                '''
                file_name = os.path.splitext(os.path.basename(each_img_file))[0] # grab the image name 
                if not result_polygons: 
                    print(f'No Polygon Found At TTA Step {step+1}. Continue...')
                    continue
                    
                if not os.path.exists(f'{args.text_output}/{file_name}'): 
                    os.makedirs(f'{args.text_output}/{file_name}') # if not there, create a new folder acc to the image name
                    
                # save the the prediction in a text format 
                # it contains: polygons, transcription, scores
                for z in zip(result_polygons, result_words, each_poly_score):
                    with open(f'{os.getcwd()}/{args.text_output}/{file_name}/{step+1}.txt', 'a') as f1:
                        f1.write(",".join(str(x) for x in z[0]) + ',' + str(z[1])+',' + str(z[2]) + '\n')
                        f1.close()


                # same as the above text annotation saver, but this time in JSON format.
                preds.append({
                    each_img_file : {
                        "prediction": result_polygons,
                        "trancription": result_words,
                        "score": each_poly_score
                    }
                })
                
                # same as text annotation saver, 
                # create a new directory (if not there) to save JSON, achieved from per step of TTA
                if not os.path.exists(f'{args.json_output}/{file_name}'): 
                    os.makedirs(f'{args.json_output}/{file_name}')
                # and save accordingly 
                json.dump(preds,open(f"{args.json_output}/{file_name}/{step + 1}.json",'w'))
                
                # visualize the outcomes from each TTA 
                # by saving respected directory
                if not os.path.exists(f'{args.visu_path}/{file_name}'):
                    os.makedirs(f'{args.visu_path}/{file_name}')
                self.visualization(image, result_polygons, result_words)
                cv2.imwrite(f'{args.visu_path}/{file_name}/{step + 1}.jpg', image) 

            # -----------------------------------------------
            # -----------------------------------------------
            # grab all text file from TTA over each sample 
            if image is None: continue
            labels = [gt for gt in sorted(glob.glob(f'{args.text_output}/{file_name}/*.txt'))]
            # hard-coded: we've used four tta step, so that
            # get_cords <- from tta_utils.py scripts
            print()
            if not labels or len(labels) > 4: 
                print(len(labels))
                print('NO TEXT ANNOTATION FOUND IN ANY TRANSFORMATION STEP. Continue...')
                continue
            elif len(labels) == 1:
                print('A Single Test Annotation Found After All TTA Step.')
                data_one, xx, yy, cts_one, scores_one = get_cords(labels[0], data = [], xx = [], yy = [], cts = [], scores = [])
                # zip them together 
                poly = np.array(list(zip(data_one, scores_one, cts_one)))
            elif len(labels) == 2:
                print('2 Test Annotation Found After All TTA Step.')
                data_one, xx, yy, cts_one, scores_one = get_cords(labels[0], data = [], xx = [], yy = [], cts = [], scores = [])
                data_two, xx, yy, cts_two, scores_two = get_cords(labels[1], data = [], xx = [], yy = [], cts = [], scores = [])
                # zip them together 
                poly = np.concatenate((
                    list(zip(data_one, scores_one, cts_one)),
                    list(zip(data_two, scores_two, cts_two))
                ))
            elif len(labels) == 3:
                print('3 Test Annotation Found After All TTA Step.')
                data_one, xx, yy, cts_one, scores_one = get_cords(labels[0], data = [], xx = [], yy = [], cts = [], scores = [])
                data_two, xx, yy, cts_two, scores_two = get_cords(labels[1], data = [], xx = [], yy = [], cts = [], scores = [])
                data_thr, xx, yy, cts_thr, scores_thr = get_cords(labels[2], data = [], xx = [], yy = [], cts = [], scores = [])
                # zip them together 
                poly = np.concatenate((
                    list(zip(data_one, scores_one, cts_one)),
                    list(zip(data_two, scores_two, cts_two)),
                    list(zip(data_thr, scores_thr, cts_thr))
                ))
            elif len(labels) == 4:
                print('4 Test Annotation Found After All TTA Step.')
                data_one, xx, yy, cts_one, scores_one = get_cords(labels[0], data = [], xx = [], yy = [], cts = [], scores = [])
                data_two, xx, yy, cts_two, scores_two = get_cords(labels[1], data = [], xx = [], yy = [], cts = [], scores = [])
                data_thr, xx, yy, cts_thr, scores_thr = get_cords(labels[2], data = [], xx = [], yy = [], cts = [], scores = [])
                data_fur, xx, yy, cts_fur, scores_fur = get_cords(labels[3], data = [], xx = [], yy = [], cts = [], scores = [])
                # zip them together 
                poly = np.concatenate((
                    list(zip(data_one, scores_one, cts_one)),
                    list(zip(data_two, scores_two, cts_two)),
                    list(zip(data_thr, scores_thr, cts_thr)),
                    list(zip(data_fur, scores_fur, cts_fur)))
                )

            # calling Non-Max-Supression over Polygon
            # poly_nms <- from tta_utils.py scripts
            op       = poly_nms(poly, 0.50)
            op_array = poly[op]

            # unzip the filter output array from poly_nms
            op_array_poly, op_array_score, op_array_cts = zip(*op_array)
            for z in zip(op_array_poly, op_array_cts, op_array_score):
                with open(f'{os.getcwd()}/{args.text_output}/{file_name}/result.txt', 'a') as f1:
                    f1.write(",".join(str(x) for x in z[0]) + ',' + str(z[1])+',' + str(z[2]) + '\n')
                    f1.close()
            
            # vis_polygons <- from tta_utils.py scripts
            vis_polygons(temp_image, op_array_poly)
            cv2.imwrite(f'{args.visu_path}/{file_name}/result.jpg', temp_image) 

            # ---------------------
            # free up some space 
            g=gc.collect()
            del image, result_polygons, result_words, each_poly_score

    def process_char_mask(self, char_masks, boxes, threshold=192):
        texts, rec_scores = [], []
        for index in range(char_masks.shape[0]):
            box = list(boxes[index])
            box = list(map(int, box))
            text, rec_score, _, _ = getstr_grid(char_masks[index,:,:,:].copy(), box, threshold=threshold)
            texts.append(text)
            rec_scores.append(rec_score)
        return texts, rec_scores

    def mask2polygon(self, mask, box, im_size, threshold=0.5, output_polygon=True):
        # mask 32*128
        image_width, image_height = im_size[1], im_size[0]
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        
        # [followng if statement is unofficial code] 
        if box_h == 0 or box_w == 0: return
        
        cls_polys = (mask*255).astype(np.uint8)
        poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
        poly_map = poly_map.astype(np.float32) / 255
        poly_map=cv2.GaussianBlur(poly_map,(3,3),sigmaX=3)
        ret, poly_map = cv2.threshold(poly_map,0.5,1,cv2.THRESH_BINARY)
        if output_polygon:
            SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            poly_map = cv2.erode(poly_map,SE1) 
            poly_map = cv2.dilate(poly_map,SE1);
            poly_map = cv2.morphologyEx(poly_map,cv2.MORPH_CLOSE,SE1)
            try:
                _, contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            except:
                contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(contours)==0:
                print(contours)
                print(len(contours))
                return None
            max_area=0
            max_cnt = contours[0]
            for cnt in contours:
                area=cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_cnt = cnt
            perimeter = cv2.arcLength(max_cnt,True)
            epsilon = 0.01*cv2.arcLength(max_cnt,True)
            approx = cv2.approxPolyDP(max_cnt,epsilon,True)
            pts = approx.reshape((-1,2))
            pts[:,0] = pts[:,0] + box[0]
            pts[:,1] = pts[:,1] + box[1]
            polygon = list(pts.reshape((-1,)))
            polygon = list(map(int, polygon))
            if len(polygon)<6:
                return None     
        else:      
            SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            poly_map = cv2.erode(poly_map,SE1) 
            poly_map = cv2.dilate(poly_map,SE1);
            poly_map = cv2.morphologyEx(poly_map,cv2.MORPH_CLOSE,SE1)
            idy,idx=np.where(poly_map == 1)
            xy=np.vstack((idx,idy))
            xy=np.transpose(xy)
            hull = cv2.convexHull(xy, clockwise=True)
            #reverse order of points.
            if  hull is None:
                return None
            hull=hull[::-1]
            #find minimum area bounding box.
            rect = cv2.minAreaRect(hull)
            corners = cv2.boxPoints(rect)
            corners = np.array(corners, dtype="int")
            pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
            polygon = [x * 1.0 for x in pts]
            polygon = list(map(int, polygon))
        return polygon

    # a function that will plot the image with prediction 
    def visualization(self, image, polygons, words):
        for polygon, word in zip(polygons, words):
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1,1,2))
            xmin = min(pts[:,0,0])
            ymin = min(pts[:,0,1])
            cv2.polylines(image,[pts],True,(0,0,255))
            cv2.putText(image, word, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            
def main(args):
    # update the config options with the config file
    cfg.merge_from_file(args.config_file)
    # manual override some options
    # cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    text_demo = TextDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        output_polygon=True
    )
    #load image and then run prediction
    text_demo.compute_prediction(args.image_paths, args.tta_step)


# start main function ..............
if __name__ == "__main__":
    
    # creating some essential folders
    root    = './MaskTextSpotterV3/tools/' 
    [os.makedirs(root+folder, exist_ok=True) for folder in ['out_images', 'out_text', 'out_json']]
    
    parser = argparse.ArgumentParser(description='parameters for demo')
    # config file 
    parser.add_argument("--config-file",  type=str, default='configs/mixtrain/seg_rec_poly_fuse_feature.yaml')
    # image folder
    parser.add_argument("--image_paths",  type=str, default="tools/image_directory")
    # prediciton (in a json format)
    parser.add_argument("--json_output",  type=str, default="tools/out_json")
    # prediction (in a text format)
    parser.add_argument("--text_output",  type=str, default="tools/out_text")
    # prediciton (in a visual format)
    parser.add_argument("--visu_path",    type=str, default="tools/out_images")
    # tta step number
    parser.add_argument("--tta_step",     type=int, default=4)
    
    args = parser.parse_args()
    assert args.tta_step <= 4 ,"TTA Step shouold not more than 4 times"

    main(args)