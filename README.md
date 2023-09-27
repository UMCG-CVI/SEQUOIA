# SEQUOIA
automated multiclass SEgmentation, QUantification, and visualizatiOn of the dIseased Aorta on hybrid PET/CT

![Logo](https://github.com/gvpraagh/SEQUOIA/tree/main/Images/SEQUOIA_logo.jpg)

<img src="https://github.com/gvpraagh/SEQUOIA/tree/main/Images/SEQUOIA_logo.jpg" width="128"/>

SEQUOIA is an automated pipeline for aortic PET/CT studies. First, it segments the ascending aorta, aortic arch, descending aorta, and abdominal aorta on low-dose CT from PET/CT. The multiclass segmentation is done using a 3D U-Net. Moreover, SEQUOIA provides tracer uptake values, calcium scores, background measurements, radiomic features, and a 2D visualization of the calcium and tracer distribution.

A CT scan can be used as input in niftii format. Based on this, the segmentation, calcium scores, and background measurement will be done. If the corresponding PET scan is given in niftii format too, the tracer uptake values and background measurements in PET will be given too. Make sure the PET is converted to wanted value: e.g., standardised uptake values (SUV), SUV normalized to lean body mass (SUL).

Please cite our article when you use SEQUOIA:
citation!

Folder structure

How to use
