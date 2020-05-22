# Graph-cut based Unsupervised 3D Segmentation of Femur

<div content="" clear="both" display="table">
  <a href="https://github.com/eDIMESLab">
  <div float="left" padding="5px" width="50%">
    <img src="https://avatars2.githubusercontent.com/u/58266717?s=200&v=4" width="90" height="90">
  </div>
  </a>
  <a href="https://github.com/UniboDIFABiophysics">
  <div float="left" padding="5px" width="50%">
    <img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
  </div>
  </a>
</div>



| **Authors**  | **Project** |  **Build Status** |
|:------------:|:-----------:|:-----------------:|
| [**D. Dall'Olio**](https://github.com/DanieleDallOlio) <br/> [**N. Curti**](https://github.com/Nico-Curti)  |  **3DFemurSegmentation**  | **Linux/MacOS** : [![travis](https://travis-ci.com/eDIMESLab/3DFemurSegmentation.svg?branch=master)](https://travis-ci.com/eDIMESLab/3DFemurSegmentation) <br/> **Windows** : [![appveyor](https://ci.appveyor.com/api/projects/status/g3wjvsf4eqo6ts96?svg=true)](https://ci.appveyor.com/project/Nico-Curti/3dfemursegmentation) |

Make sure to install conda before going further.

Clone the repository:
```console
username@local:~$ git clone https://github.com/eDIMESLab/3DFemurSegmentation.git
```

Next, in order to create 3DFemurSegmentation custom environment, type:
```console
username@local:~/3DFemurSegmentation$ conda env create -f itk.yaml
```

Then, activate the environment and build Cython libraries:
```console
username@local:~/3DFemurSegmentation$ conda activate itk
username@local:~/3DFemurSegmentation$ python setup.py develop --user
```

Make sure to add `~/3DFemurSegmentation/lib/` to your Python library path before running. On Ubuntu OS, type:

```console
username@local:~/3DFemurSegmentation$ export PYTHONPATH=$PYTHONPATH:~/3DFemurSegmentation/lib/
```

Now, run the unsupervised segmentation by supplying input DICOMs directory and output DICOMs directory:

```console
username@local:~/3DFemurSegmentation$ python runFemurSegmentation.py --indir ./indir_example  --outdir ./outdir_example
```

## Authors

* **Daniele Dall'Olio** [git](https://github.com/DanieleDallOlio), [unibo](https://www.unibo.it/sitoweb/daniele.dallolio)
* **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)

## Acknowledgments

Thanks goes to all contributors of this project.

## References

<a id="1">[1]</a>
Krčah, M., Székely, G., Blanc, R.
Fully automatic and fast segmentation of the femur bone from 3D-CT images with no shape prior
2011 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, Chicago, IL, 2011, pp. 2087-2090. [doi](https://doi.org/10.1109/ISBI.2011.5872823)
