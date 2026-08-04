Mesa 3D Custom
--------

This repository contains customized versions of the Mesa drivers used in Winlator.
The original source code lives at https://gitlab.freedesktop.org/mesa/mesa.

Build Mesa Turnip
--------

Navigate to the source folder and run the following commands:

```
$ meson build -Dstrip=true -Dcpp_rtti=false -Dgbm=disabled -Dopengl=false -Dllvm=disabled -Dshared-llvm=disabled -Dplatforms=x11 -Dgallium-drivers= -Dxmlconfig=disabled  -Dvulkan-drivers=freedreno -Dmicrosoft-clc=disabled -Dlibunwind=disabled -Dvalgrind=disabled -Dfreedreno-kmds=kgsl,msm -Dprefix=/data/data/com.winlator/files/rootfs/usr -Dbuildtype=release
$ ninja -j8 -C build
```

Build Mesa Zink
--------

Navigate to the source folder and run the following commands:

```
$ meson build -Dstrip=true -Dplatforms=x11 -Dgallium-va=disabled -Ddri-drivers= -Dllvm=disabled -Dshared-llvm=disabled -Dgallium-drivers=zink -Ddri3=disabled -Dvulkan-drivers= -Dglx=xlib -Dlmsensors=disabled -Dprefix=/data/data/com.winlator/files/rootfs/usr -Dbuildtype=release
$ ninja -j8 -C build
```

Build Mesa VirGL
--------

Navigate to the source folder and run the following commands:

```
$ meson build -Dstrip=true -Dgallium-va=disabled -Dgallium-drivers=virgl -Ddri3=disabled -Dllvm=disabled -Dshared-llvm=disabled -Dvulkan-drivers= -Dglx=xlib -Dplatforms=x11 -Dlmsensors=disabled -Dprefix=/data/data/com.winlator/files/rootfs/usr -Dbuildtype=release
$ ninja -j8 -C build
```