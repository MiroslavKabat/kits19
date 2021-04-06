# kits19 🎗️
Demonstration how to get kits19 intersection over union (IOU) above 80 % with 2D UNet using transfer learning!

| Raw image | Mask (Ground Truth) | Prediction |
|:---:|:---:|:---:|
| ![GitHub Logo](/sample/case_00123_172.png) |  ![GitHub Logo](/sample/case_00123_172_mask_kid.png) | ![GitHub Logo](/sample/case_00123_172_prediction.png) |

# Quick start 🎬
* Read challange paper: [kits19.grand-challenge.org](https://kits19.grand-challenge.org/home/)
* Follow neheller's github to get data a libraries: [github](https://github.com/neheller/kits19)
* If you have downloaded `data` folder including `imaging.nii.gz` and `segmentation.nii.gz` in each folder (first 209 only), you can start using my scripts
    * Run `sanitycheck.py` to create `png` folder with visualisation of raw data
    * Run `convertniitonpz.py` to create `npz` folder with preprocessed images compressed in to `*.npz` files for fast data loading
    * Run `train.py` to create `models` folder with timestamps folders containing TensorBoard, .csv and models!
    * Run `predict.py` to create `models/{timestamp}/predictions` with predicted heatmaps in grayscale

## Contribution 🤝
Pull Request welcome

## Contact 🤙🏻
If you need help with your project, my work helps you or you have any ideas how to improve my code, let me know about it!

😁 www.MiroslavKabat.com

✉️ hello@miroslavkabat.com

## Donations ❤️
I spent in this project more like 🕒 16 man-hours with ⌛ 18 months of getting knowledge and 💸 6000 $ for workstation, I appreciate any donations.
[PayPal Donations](https://www.paypal.com/donate?hosted_button_id=36V6T4WK7W5NS) Thank you! 
