from util.run_ucmc import run_ucmc, make_args

if __name__ == '__main__':

    det_path = "det_results/mot20"
    cam_path = "cam_para/MOT20"
    gmc_path = "gmc/mot20"
    out_path = "output/mot20"
    exp_name = "val"
    dataset = "MOT20"
    args = make_args()

    run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name,dataset)