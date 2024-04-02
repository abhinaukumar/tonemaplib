# TonemapLIB
_HDR Tone Mapping in Python._

# Description
The TonemapLIB library provides an easy API for tone mapping high dynamic range (HDR) videos. TonemapLIB is built on [VideoLIB](https://github.com/abhinaukumar/videolib), and was used to generate videos for the [LIVE-TMHDR dataset](https://live.ece.utexas.edu/research/LIVE_TMHDR/index.html). TonemapLIB provides the following salient features.

1. A standardized `TMO` class that defines tone mapping operator (TMO) behavior.
2. Options to specify and vary TMO arguments using `.py` files (refer to the `tmo_args/` folder for examples).
3. Tone Mapping in three temporal modes - _framewise_, _shot_, and _smooth_. The three modes are described in detail in [this paper](https://arxiv.org/abs/2403.15061).
4. Open-source implementations of ten TMOs from the literature. 
    - Reinhard02 [[Paper]](https://dl.acm.org/doi/abs/10.1145/2816795.2818092) [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/reinhard02.py)
    - Durand02 [[Paper]](https://dl.acm.org/doi/abs/10.1145/566570.566574) [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/durand02.py)
    - Hable [[Paper]](https://www.gdcvault.com/play/1012351/Uncharted-2-HDR) [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/hable.py)
    - Shan12 [[Paper]](https://graphics.pixar.com/library/ToneMappingVideoUsingWavelets/paper.pdf)  [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/shan12.py)
    - Reinhard12 [[Paper]](https://d1wqtxts1xzle7.cloudfront.net/38140358/SASIA2012-libre.pdf?1436451899=&response-content-disposition=inline%3B+filename%3DCalibrated_Image_Appearance_Reproduction.pdf&Expires=1712087261&Signature=BdMm9LI~zU706v73An0zdqunAjIQ2vtjVxSXN6nIsRZqC3rTTjfw83Y~k9JpmcmZ8vo7tEseWxaJysgvIXWh1j8ahfLAz~TsgzPYq~28-c5yGogkdaBLgkNqgG2k1vfMsNgOPT~Bai5xZQ8U4S1mfJSI7lr0fsII0cj2fFiH2GRd1H3YJ4rHDmlVZHEx9ttGg8GjS7OlelO0toPZVvoobUlCb52LZX9Rg6Iayomar9o7tcUvnvbtQLknFeUSHmVDpn4wnDNbFIyJiWYUOpP-uSiT0xmO7G1ujbs091MnTeEi6SJiRshoXYP8sriwXGP7ssNpNx548awXK6tNtxSeFA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/reinhard12.py)
    - Eilertsen15 [[Paper]](https://dl.acm.org/doi/abs/10.1145/2816795.2818092) [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/eilertsen15.py)
    - Oskarsson17 [[Paper]](https://link.springer.com/article/10.1007/s10851-016-0677-1) [[Paper]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/oskarsson17.py)
    - Rana19 [[Paper]](https://ieeexplore.ieee.org/abstract/document/8822603) [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/rana19.py)
    - Yang21 [[Paper]](https://arxiv.org/abs/2102.00348) [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/yang21.py)
    - ITU21 [[Paper]](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2446-1-2021-PDF-E.pdf) [[Code]](https://github.com/abhinaukumar/tonemaplib/blob/main/tonemaplib/itu21.py)

_Note: The implementations may not be identical to their descriptions in the cited papers._

# Usage
The usage example provided below offers a glimpse into the degree to which VideoLIB simplifies video processing tasks.

## Tone Mapping an HDR Video
```python
from videolib import Video, standards
from tonemaplib import get_tmoclass

TMOClass = get_tmoclass('Reinhard02')
tmo = TMOClass()
with Video('hdr_video_path.mp4', standards.rec_2100_pq, 'r') as v:
    tmo(v, 'sdr_video_path.mp4')
```

## Setting TMO parameters
```python
from videolib import Video, standards
from tonemaplib import get_tmoclass

TMOClass = get_tmoclass('Reinhard02')
print(TMOClass.params)
tmo = TMOClass(key=0.2, desat=0.01, video_mode='shot')
with Video('hdr_video_path.mp4', standards.rec_2100_pq, 'r') as v:
    tmo(v, 'sdr_video_path.mp4')
```


# Installation
To use TonemapLIB, you will need Python >= 3.7.0. In addition, using virtual environments such as `virtualenv` or `conda` is recommended. The code has been tested on Linux and it is expected to be compatible with Unix/MacOS platforms.

## Creating a virtual environment
```bash
python3 -m virtualenv .venv/
source .venv/bin/activate
```
## Install preqrequisites and TonemapLIB
```bash
pip install -r requirements.txt
pip install .
```
## Install using pip
```bash
pip install git+github.com/abhinaukumar/tonemaplib
```

# Issues, Suggestions, and Contributions
Please [file an issue](https://github.com/abhinaukumar/tonemaplib/issues) if you would like to suggest a feature, or flag any bugs/issues, and I will respond to them as promptly as I can. Contributions that add features and/or resolve any issues are also welcome! Please create a [pull request](https://github.com/abhinaukumar/tonemaplib/pulls) with your contribution and I will review it at the earliest.

# Contact Me
If you would like to contact me personally regarding TonemapLIB, please email me at either [abhinaukumar@utexas.edu](mailto:abhinaukumar@utexas.edu) or [ab.kumr98@gmail.com](mailto:ab.kumr98@gmail.com).

# License
TonemapLIB is covered under the MIT License, as shown in the [LICENSE](https://github.com/abhinaukumar/tonemaplib/blob/main/LICENSE) file.
