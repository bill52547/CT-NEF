!/bin/sh
#
cp $NEFCT_USER_OP_SOURCE/BUILD .
cp -rf $NEFCT_USER_OP_SOURCE/siddon_2d .
cp -rf $NEFCT_USER_OP_SOURCE/siddon_3d .
cp -rf $NEFCT_USER_OP_SOURCE/ray_2d .
cp -rf $NEFCT_USER_OP_SOURCE/ray_3d .
cp -rf $NEFCT_USER_OP_SOURCE/pixel_2d .
cp -rf $NEFCT_USER_OP_SOURCE/pixel_3d .
# cp -rf $NEFCT_USER_OP_SOURCE/deform .

#(($TF_USER_OP_ROOT == $(pwd));
#if [ $TF_USER_OP_ROOT -ne $(pwd) ]; then echo "error: must run in a bazel workspace"

#cp -rf ${NEF_USER_OP_SOURCE}/* .

#cd $TF_USER_OP_ROOT

bazel build -c opt //tensorflow/core/user_ops:ct_siddon2d_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures
bazel build -c opt //tensorflow/core/user_ops:ct_siddon3d_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures
bazel build -c opt //tensorflow/core/user_ops:ct_ray2d_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures
bazel build -c opt //tensorflow/core/user_ops:ct_ray3d_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures
bazel build -c opt //tensorflow/core/user_ops:ct_pixel2d_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures
bazel build -c opt //tensorflow/core/user_ops:ct_pixel3d_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures

# bazel build -c opt //tensorflow/core/user_ops:tf_deform_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures