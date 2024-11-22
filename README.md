# Invariant Learning Improves Out-of-Distribution Generalization for IP Geolocation
This repository provides the original PyTorch implementation of the GELGeo framework.

## Process data
```
cd lib
python preprocess.py
```

## Run examples
```
python run.py
```

## Dataset Information

The "datasets" folder contains three subfolders corresponding to three large-scale real-world street-level IP geolocation    datasets collected from New York City, Los Angeles, and Shanghai. There are three files in each subfolder:

- data.csv    *# features (including attribute knowledge and network measurements) and labels (longitude and latitude) for street-level IP geolocation* 
- ip.csv    *# IP addresses*
- last_traceroute.csv    *# last four routers and corresponding delays for efficient IP host clustering*

The detailed **columns and description** of data.csv in the New York dataset are as follows:

#### New York  

| Column Name                     | Data Description                                             |
| ------------------------------- | ------------------------------------------------------------ |
| ip                              | The IPv4 address                                             |
| as_mult_info                    | The ID of the autonomous system where IP locates             |
| country                         | The country where the IP locates                             |
| prov_cn_name                    | The state/province where the IP locates                      |
| city                            | The city where the IP locates                                |
| isp                             | The Internet Service Provider of the IP                      |
| vp900/901/..._ping_delay_time   | The ping delay from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._trace             | The traceroute list from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._tr_steps          | #steps of the traceroute from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._last_router_delay | The delay from the last router to the IP host in the traceroute list from probing hosts "vp900/901/..." |
| vp900/901/..._total_delay       | The total delay from probing hosts "vp900/901/..." to the IP host |
| longitude                       | The longitude of the IP (as the label)                           |
| latitude                        | The latitude of the IP host (as the label)                       |

PS: The detailed columns and description of data.csv in the other two datasets are similar to the New York dataset.

# Requirements
- python = 3.8.13
- pytorch = 1.12.1
- cudatoolkit = 11.6.0
- cudnn = 7.6.5
