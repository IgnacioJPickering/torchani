cd /home/ignacio/Git-Repos/torchani/tools || exit 1
python ./md_scaling_benchmark.py --model ani1x_one --file-name ani1x_one_rtx 
python ./md_scaling_benchmark.py --model ani1x_all --file-name ani1x_all_rtx
python ./md_scaling_benchmark.py --model ani2x_one --file-name ani2x_one_rtx
python ./md_scaling_benchmark.py --model ani2x_all --file-name ani2x_all_rtx
python ./md_scaling_benchmark.py --model ani1ccx_one --file-name ani1ccx_one_rtx
python ./md_scaling_benchmark.py --model ani1ccx_all --file-name ani1ccx_all_rtx
