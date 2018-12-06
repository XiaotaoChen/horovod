here is the original [Installation instructions](https://github.com/ctcyang/incubator-mxnet/tree/horovod_feature/example/image-classification/scripts)

**My instructions as below.**

1. building latest MXNet on cpu (with openblas or mkldnn);

2. package header files needed by horovod to `python/mxnet/include` directory , as below ;

   ```shell
   mkdir python/mxnet/include
   cp -r ~/mxnet/3rdparty/mshadow/mshadow ~/mxnet/python/mxnet/include
   cp -r ~/mxnet/3rdparty/dlpack/include/dlpack ~/mxnet/python/mxnet/include
   cp -r ~/mxnet/3rdparty/dmlc-core/include/dmlc ~/mxnet/python/mxnet/include
   cp -r ~/mxnet/3rdparty/tvm/include/tvm ~/mxnet/python/mxnet/include
   cp -r ~/mxnet/3rdparty/tvm/nnvm/include/nnvm ~/mxnet/python/mxnet/include
   cp -r ~/mxnet/include/mxnet ~/mxnet/python/mxnet/include
   cp ~/mxnet/3rdparty/mkldnn/include/* ~/mxnet/python/mxnet/include
   ```

3. building horovod

   add `MSHADOW_USE_CUDA` and `MXNET_USE_MKLDNN` macros used by included mxnet's headed files 

   ```shell
   git clone https://github.com/XiaotaoChen/horovod.git -b mxnet-hvd-lite horovod
   source your python env with MXNet
   
   PATH=/usr/local/bin:$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 MSHADOW_USE_CUDA=0 MXNET_USE_MKLDNN=1 INCLUDES=['mxnet git directory'/python/mxnet/include] LIBRARY_DIRS=['mxnet git directory'/lib] /home/ubuntu/anaconda3/bin/pip install --upgrade -v --no-cache-dir ['horovod git directory']
   ```

4. running benchmark script according to  [Installation instructions](https://github.com/ctcyang/incubator-mxnet/tree/horovod_feature/example/image-classification/scripts)

**encounter problems with mxnet.hvd on cpu**

1. Enabled multiple instances training with mxnet.hvd on cpu when MXNet compiled with openblas;
2. when MXNet compiled with mkldnn, it only supports single instance training with mxnet.hvd. 
3. `mxnet-hvd-lite-debug`  branch can print some info for debugging;