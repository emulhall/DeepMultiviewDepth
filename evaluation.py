import numpy as np
import pathlib
from PIL import Image
import cv2
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import openpyxl

# code from "Deep Multi-view Depth Estimation with Predicted Uncertainty" modified to operate on np arrays
def eval(depths_gt,pred_depths):

    depth_mask_np = (depths_gt > 0)
    depth_ratio_np1 = (depths_gt / (pred_depths+0.00001))
    depth_ratio_np2 = (pred_depths / (depths_gt + 0.00001))

    # if self.args.uncertainty_threshold > 0.0:
    #     pred_uncertainties = self._get_network_output_uncertainty(cnn_outputs)
    #     depth_mask = (depths_gt > 0.1).float() * \
    #                  (depths_gt < 10.0).float() * \
    #                  (torch.exp(pred_uncertainties[-1]) < self.args.uncertainty_threshold).float()

    # depth_mask_np = depth_mask.detach().cpu().numpy() > 0

    all_abs_rel, all_sq_rel, all_log_rmse, all_i_rmse, all_si_log = [], [], [], [], []
    all_mad, all_rmse, all_a05, all_a10, all_a1, all_a2, all_a3 = [], [], [], [], [], [], []

    depth_abs_error_no_mask = np.abs(depths_gt - pred_depths)
    depth_ratio_error_no_mask1 = depth_ratio_np1
    depth_ratio_error_no_mask2 = depth_ratio_np2
    for i in range(depths_gt.shape[0]):
        if np.sum(depth_mask_np[i,:,:]) > 0:
            depth_abs_error_image_i = depth_abs_error_no_mask[i][depth_mask_np[i]]
            depth_ratio_error_image_i1 = depth_ratio_error_no_mask1[i][depth_mask_np[i]]
            depth_ratio_error_image_i2 = depth_ratio_error_no_mask2[i][depth_mask_np[i]]

            # additional metrics
            pr = pred_depths[i, ...][depth_mask_np[i]]
            gt = depths_gt[i, ...][depth_mask_np[i]]

            abs_rel = np.mean(np.abs(gt - pr) / gt)
            sq_rel = np.mean(((gt - pr) ** 2) / gt)
            rmse_log = np.sqrt(mean_squared_log_error(gt,pr))
            # rmse_log = (np.log(gt) - np.log(pr)) ** 2
            # rmse_log = np.sqrt(rmse_log.mean())
            i_rmse = np.sqrt(mean_squared_error(1/(gt+0.000001),1/(pr+0.000001)))
            # i_rmse = (1 / gt - 1 / (pr + 1e-4)) ** 2
            # i_rmse = np.sqrt(i_rmse.mean())

            # sc_inv
            log_diff = np.log(gt+0.000001) - np.log(pr+0.000001)
            num_pixels = np.float32(log_diff.size)
            sc_inv = np.sqrt(
                np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(
                    num_pixels))

            all_abs_rel.append(abs_rel)
            all_sq_rel.append(sq_rel)
            all_log_rmse.append(rmse_log)
            all_i_rmse.append(i_rmse)
            all_si_log.append(sc_inv)

            all_mad.append(np.mean(depth_abs_error_image_i).item())
            all_rmse.append(np.sqrt(np.mean(depth_abs_error_image_i ** 2)).item())

            all_a05.append(100. * max(np.sum(depth_ratio_error_image_i1 < 1.05),np.sum(depth_ratio_error_image_i2<1.05)) / depth_ratio_error_image_i1.shape[0])
            all_a10.append(
                100. * max(np.sum(depth_ratio_error_image_i1 < 1.10).item(),np.sum(depth_ratio_error_image_i2 < 1.10).item()) / depth_ratio_error_image_i1.shape[0])
            all_a1.append(
                100. * max(np.sum(depth_ratio_error_image_i1 < 1.25).item(),np.sum(depth_ratio_error_image_i2 < 1.25).item()) / depth_ratio_error_image_i1.shape[0])
            all_a2.append(100. * max(np.sum(depth_ratio_error_image_i1 < 1.25 ** 2).item(),np.sum(depth_ratio_error_image_i2 < 1.25 ** 2).item()) /
                          depth_ratio_error_image_i1.shape[0])
            all_a3.append(100. * max(np.sum(depth_ratio_error_image_i1 < 1.25 ** 3).item(),np.sum(depth_ratio_error_image_i2 < 1.25 ** 3).item()) /
                          depth_ratio_error_image_i1.shape[0])

    depth_error_every_image = {'abs_rel': np.array(all_abs_rel),
                               'sq_rel': np.array(all_sq_rel),
                               'rmse_log': np.array(all_log_rmse),
                               'i_rmse': np.array(all_i_rmse),
                               'sc_inv': np.array(all_si_log),
                               'mad': np.array(all_mad),
                               'rmse': np.array(all_rmse),
                               '1.05': np.array(all_a05),
                               '1.1': np.array(all_a10),
                               '1.25': np.array(all_a1),
                               '1.25^2': np.array(all_a2),
                               '1.25^3': np.array(all_a3)}
    depth_error_avg = {'abs_rel': np.average(np.array(all_abs_rel)),
                               'sq_rel': np.average(np.array(all_sq_rel)),
                               'rmse_log': np.average(np.array(all_log_rmse)),
                               'i_rmse': np.average(np.array(all_i_rmse)),
                               'sc_inv': np.average(np.array(all_si_log)),
                               'mad': np.average(np.array(all_mad)),
                               'rmse': np.average(np.array(all_rmse)),
                               '1.05': np.average(np.array(all_a05)),
                               '1.1': np.average(np.array(all_a10)),
                               '1.25': np.average(np.array(all_a1)),
                               '1.25^2': np.average(np.array(all_a2)),
                               '1.25^3': np.average(np.array(all_a3))}
    return depth_error_every_image, depth_error_avg

def main():
    filename = 'waving5'

    pred_loc = r'E:\Python\CSCI5563Homework\venv\Project\completed vids'
    # pred_depths = np.array([np.asarray(Image.open(path))for path in pathlib.Path(rf'{pred_loc}\{filename}').iterdir()])
    # imsize = pred_depths.shape[1:]
    vid = cv2.VideoCapture(rf'{pred_loc}\{filename}_depth_baseline.mp4')
    imlist = []
    while vid.isOpened():
        success, image = vid.read()
        if success:
            imlist.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        else:
            break
    cv2.destroyAllWindows()
    vid.release()
    # get size of image
    imsize = imlist[0].shape

    pred_depths = np.array(imlist)
    mins = np.amin(pred_depths,axis=(1,2))
    maxs = np.amax(pred_depths, axis=(1,2))
    scaled_pred = np.array([(pred_depths[ii,:,:]-mins[ii])/maxs[ii] for ii,_ in enumerate(pred_depths)])

    # get ground truth images from folder
    gt_loc = r'E:\Python\CSCI5563Homework\venv\Project\output'
    depths_gt = np.array([cv2.resize(np.asarray(Image.open(path)),imsize[::-1]) for path in pathlib.Path(rf'{gt_loc}\{filename}_depth').iterdir()])
    np.save('shifted_scaled_waving5',scaled_pred)
    # wierd pred_depths is to sync predictions with ground truth frames
    all_result, avg_result = eval(scaled_pred[-depths_gt.shape[0]:, :, :],
                                  depths_gt[0:pred_depths.shape[0], :, :] / 255)
    # all_result,avg_result = eval(scaled_pred[-depths_gt.shape[0]:,None,:,:],depths_gt[0:pred_depths.shape[0],None,:,:]/255)
    # all_result,avg_result = eval(scaled_pred,depths_gt/255)
    print(avg_result)

    # workbook = openpyxl.load_workbook('baseline_result.xlsx')
    # sheet = workbook.active
    # last_row = sheet.max_row
    # sheet.append([filename, avg_result['abs_rel'],avg_result['sq_rel'],avg_result['rmse_log'],avg_result['i_rmse'],
    #               avg_result['sc_inv'],avg_result['mad'],avg_result['rmse'],avg_result['1.05'],avg_result['1.1'],
    #               avg_result['1.25'],avg_result['1.25^2'], avg_result['1.25^3']])
    # workbook.save('baseline_result.xlsx')


if __name__=='__main__':
    result = main()
