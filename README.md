# Fit Mocap C3D into SMPL-X

The primary purpose of this project is to fit .c3d file into a SMPL-X model. 

Based on the statement and link in [AMASS dataset](https://github.com/nghorbani/amass):

```bash
AMASS uses MoSh++ pipeline to fit SMPL+H body model to human optical marker based motion capture (mocap) data. In the paper we use SMPL+H with extended shape space, i.e. 16 betas, and 8 DMPLs. Please download models and place them them in body_models folder of this repository after you obtained the code from GitHub.
```

I ended up with [SOMA](https://github.com/nghorbani/soma). However, the whole process of environment configuration is vague and full of outdated information.

So I tried my best to figure out the configuration of conda, running through the tutorials in [SOMA tutorial](https://github.com/nghorbani/soma/tree/main/src/tutorials), and how to customize my own proceeded .c3d file.

### Workstation

+ Ubuntu: 22.04
+ Nvidia driver: 12.2

### Conda environment

+ Basically you need the system dependencies and create an environment:

  ```bash
  sudo apt install libatlas-base-dev
  sudo apt install libpython3.7
  sudo apt install libtbb2
  
  conda create -n soma python=3.7 
  conda install -c conda-forge ezc3d
  ```

  The `ezc3d` package installation is currently not supported by pip.

  For my situation, `torch==1.8.2+cu102` is not working due to the error "CUDA image xxx", so I installed with:

  ```bash
  git clone https://github.com/nghorbani/soma.git
  pip install -r requirements.txt
  
  pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
  
  ```

  I didn't ever try with other combinations of `torch` and `cuda` version, so if you meet with error like me, you can debug in this way.

  Afterwards, we ended up with the installation of `soma`:

  ```bash
  python setup.py develop
  ```

+ In SOMA repo, it is instructed to first  "Copy the precompiled [smpl-fast-derivatives](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=smpl-fast-derivatives.tar.bz2) into your python site-packages folder", and then "Install the psbody.mesh library following the instructions". 

  But here I recommend to first install [psbody.mesh](https://github.com/MPI-IS/mesh) following the instructions:

  ```bash
  git clone https://github.com/MPI-IS/mesh.git
  cd mesh
  sudo apt-get install libboost-dev
  
  BOOST_INCLUDE_DIRS=/path/to/boost/ make all
  
  # Generally you can find `boost` in `/usr/include/boost`
  # For instance: BOOST_INCLUDE_DIRS=/usr/include/boost make all
  
  python setup.py install
  ```

  You will have a `psbody` package in your path, i.e. `anaconda3/envs/soma/lib/python3.7/site-packages/psbody/mesh`

  Afterwards, we copy the precompiled [smpl-fast-derivatives](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=smpl-fast-derivatives.tar.bz2) in this path and the final directory should be look like:

  ```bash
  <path to env>/lib/python3.7/site-packages/
  ├─ psbody/
  │  ├─ mesh/
  │  └─ smpl/
  ```

  Now you can test the installation in python:

  ```python
  from psbody import smpl
  from psbody import mesh	
  ```

  ###### Note:

  Every time we modify the content in `mesh` we should repeat the process of `make` and `python setup.py install`.

+ There is no problem for the Blender-2.83 LTS and [bpy-2.83](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=blender/bpy-2.83-20200908.tar.bz2) part described in SOMA repo, we can just download by snap and then uncompress contents of the precomplied bpy-2.83 into package folder.

+ Ultimately, we arrived the final step of installing [Mosh++](https://github.com/nghorbani/moshpp). Fortunately it is simple to follow the instruction:

  ```bash
  sudo apt install libtbb-dev
  
  gitc clone https://github.com/nghorbani/moshpp.git
  cd moshpp
  
  # you first need to delete the 'sklearn' in the requirements
  pip install -r requirements.txt
  
  cd src/moshpp/scan2mesh
  sudo apt install libeigen3-dev
  pip install -r requirements.txt
  cd mesh_distance
  make
  
  cd ../../../..
  python setup.py install
  ```

**Now we have installed the whole environment !!!**

### The tutorial in SOMA

###### Note

It is highly recommend to read about the [Tutorial Readme in SOMA](https://github.com/nghorbani/soma/blob/main/src/tutorials) and follow up the origin tutorial in case you don't know what we are going through and what we are talking about!

Basically you need to download the template folder and the corresponding but here I have created the directory `workspace` and push part of data.

Here I only tried [The first tutorial](https://github.com/nghorbani/soma/blob/main/src/tutorials/run_soma_on_soma_dataset.ipynb) in SOMA because it is enough for me. You also need to follow the step of:

```python
1.Solving SOMA MoCap Dataset
2.SOMA MoCap Dataset
3.Prepare Body Dataset For Training
4.Prepare Body Model and Co.
```

Afterward, we get our workspace directory look like: []("https://download.is.tue.mpg.de/soma/tutorials/tutorial_training_folder_structure.png")

#### Training SOMA

In the first code block you first change the `soma_work_base_dir` to your path, i.e.

```python
soma_expr_id = 'V48_02_SOMA'

soma_data_settings = [(5, 3, 0.0, 1.0), ] # upto 5 occlusions, upto 3 ghost points, 0.0% real data, 100. % synthetic data
soma_work_base_dir = '/home/ubuntu/Desktop/test/workspace'
support_base_dir = osp.join(soma_work_base_dir, 'support_files')
soma_marker_layout_fname = osp.join(support_base_dir, 'marker_layouts/SOMA/soma_subject1/clap_001.c3d')

num_gpus = 1 # number of gpus for training
num_cpus = 4
```

In the `train_multiple_soma` block you might meet the first error:

```bash
TypeError: __init() got an unexcepted keyword argument 'distributed_backend'
```

The solution is to replace `distributed_backend` in `soma/support_data/conf/soma_train_conf.yaml` and `soma_trainer.py--line 377` with `strategy`.

After modification, the next RunTimeError is related to the `torch.use_deterministic_algorithm(True)`. However, we cannot directly set `torch.use_deterministic_algorithm(True, warn_only=True)` because of the version of `torch`. Here I simply set `deterministic: false` in `soma_train_conf.yaml--line 125`.

Up to now, the `train_multiple_soma` block is completed!

#### Running SOMA On Mocap Point Cloud Data

In `run_soma_on_multiple_settings` you might meet 

```bash
UnsupportedInterpolationType: Unsupported interpolation type resolve_mocap_subject
    full_key: dirs.mocap_out_fname
    object_type=dict
```

The solution is to replace all `resolve_mocap_subject` with `resolve_mocap_session` in `soma_run_conf.yaml`, `eval_v2v.yaml`, and `eval_label.yaml`.

Because I don't care about the experiment results in the original paper, so I ignored the related blocks with key `ds_name_gt` in calling functions. 

*PS: you can try with downloading the labled data by yourself.*

#### Solving Bodies with Mosh++

You can directly run the block without error and as the same, I skipped the `ds_name_gt` parts.

#### Rendering Solved Bodies

If you have installed Blender 2.83-LTS following the installation instructions you can also render solved bodies using Blender. Download the [Blender blend files](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=blender/blend_files.tar.bz2) and place them under `support_files/blender/blend_files`.

The rendering part is correct but if you want to use the AMASS tutorials and the body visualizer to turn mosh pkl files into AMASS npz format on the fly and render in Jupyter, there is some problem.

It is mentioned `No module named pyrender` but the whole conda env cannot handle the conflict between `PyOpenGL`, `pyglet`, and `pyrender`.  If you install `pyrender==0.1.45`, it will degrade the `PyOpenGL` to 3.1.0, after which, there will be an error about `GLUT` when you re-run the Jupyter.

So here I write `enable_visualize.sh` and `disable_visualize.sh` in `workspace/scripts` for convenience.

For now, we have passed through the whole `run_soma_on_soma_dataset.ipynb` !

### Customize your own mocap 

When you have your own .c3d file of optical marker, you can fit it into a customized SMPL-X model for downstream tasks.

Here in `workspace/scripts/mocap2smpl.py` you should change the `soma_mocap_target_ds_name` to yours and create the corresponding folder.

Detailly:

+ I implemented `gen_stagei_mocap_fnames_customized` in `soma.run_soma.paper_plots.mosh_soma_dataset`

+ ```python
  write_mocap_c3d(out_mocap_fname=c3d_out_fname,
                        markers=results['markers'],
                        labels=nan_replaced_labels,
                        frame_rate=soma_labeler.mocap_frame_rate)
  ```
  in `soma_processor.py--line 451`

+ add

  ```python
  frame_picker_cfg.num_frames = min(frame_picker_cfg.num_frames, len(stage1_mocap_fnames))
  ```

  in `moshpp/mosh_head.py--line 137`. 

+ modify the config for start and end index to be moshed in `moshpp/chmosh.py--line 539` and `python setup.py install` for Mosh++.

For visualization:

```bash
zsh enable_visualize.sh
python visualize.py
zsh disable_visualize.sh
```

### Demo

the original video is downloaded from [KIT Motions](https://motion-database.humanoids.kit.edu/details/motions/3130/?listpage=3) and the processed video of SMPL-X are showed at:
`workspace/scripts/sweep.mp4`

# Note:

***I have made minimum modifications to the original code only for my stake so it may be a little bit ugly and hardcode. If you want to help me beautify it, feel appreciate to contact with me !***