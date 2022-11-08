#include "utils.h"

#ifdef __VITIS_CL__
void initialize_device(cl_object &obj) {
    std::cout << "Initializing device...\n" << std::endl;

    std::vector<cl::Device> devices;
    cl_int err;

    std::vector<cl::Platform> platforms;
    bool found_device = false;

    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false); i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                obj.device = devices[0];
                found_device = true;
                break;
            }
        }
    }

    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device "
                << obj.device.getInfo<CL_DEVICE_NAME>() << std::endl;
        exit(EXIT_FAILURE);
    }

    // Creating Context and Command Queue for selected device
    OCL_CHECK(err, obj.context =
            cl::Context(obj.device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, obj.q = cl::CommandQueue(obj.context, obj.device,
            CL_QUEUE_PROFILING_ENABLE, &err));
}

void read_xclbin(std::string xclbinFilename, cl::Program::Binaries &bins) {
    std::cout << "INFO: Reading " << xclbinFilename << std::endl;

    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n",
                xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }

    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);

    bins.push_back({buf,nb});
}

void program_kernel(cl_object &cl_obj, krnl_object &krnl_obj) {
    cl_int err;

    if (cl_obj.program != NULL) {
        delete cl_obj.program;
        delete cl_obj.krnl;

        cl_obj.program = NULL;
        cl_obj.krnl = NULL;
    }

    std::cout << "Programming kernel " << krnl_obj.name << "..." << std::endl;
    OCL_CHECK(err, cl_obj.program = new cl::Program(cl_obj.context,
            {cl_obj.device}, {cl_obj.bins[krnl_obj.index]}, NULL, &err));

    // This call will get the kernel object from program. A kernel is an
    // OpenCL function that is executed on the FPGA.
    OCL_CHECK(err, cl_obj.krnl =
            new cl::Kernel(*cl_obj.program, krnl_obj.name.c_str(), &err));
}
#endif

// Allocate memory on device and map pointers into the host
void allocate_readonly_mem(cl_object &cl_obj, void **ptr, uint64_t idx,
                           uint64_t size_in_bytes) {
#ifdef __VITIS_CL__
    cl_int err;

    // These commands will allocate memory on the Device. The cl::Buffer
    // objects can be used to reference the memory locations on the device.
    OCL_CHECK(err, cl_obj.buffers.emplace_back(cl_obj.context,
            CL_MEM_READ_ONLY, size_in_bytes, nullptr, &err));

    cl::Buffer *buffer = &cl_obj.buffers[idx];

    //We then need to map our OpenCL buffers to get the pointers
    OCL_CHECK(err, *ptr = (void*) cl_obj.q.enqueueMapBuffer (*buffer, CL_TRUE,
            CL_MAP_WRITE, 0, size_in_bytes, NULL, NULL, &err));
#else
    MALLOC_CHECK(*ptr = (void*) new char[size_in_bytes])
#endif
}

void allocate_readwrite_mem(cl_object &cl_obj, void **ptr, uint64_t idx,
                            uint64_t size_in_bytes) {
#ifdef __VITIS_CL__
    cl_int err;

    // These commands will allocate memory on the Device. The cl::Buffer
    // objects can be used to reference the memory locations on the device.
    OCL_CHECK(err, cl_obj.buffers.emplace_back(cl_obj.context,
            CL_MEM_READ_WRITE, size_in_bytes, nullptr, &err));

    cl::Buffer *buffer = &cl_obj.buffers[idx];

    //We then need to map our OpenCL buffers to get the pointers
    OCL_CHECK(err, (*ptr) = (void*) cl_obj.q.enqueueMapBuffer (*buffer, CL_TRUE,
            CL_MAP_READ | CL_MAP_WRITE, 0, size_in_bytes, NULL, NULL, &err));
#else
    MALLOC_CHECK(*ptr = (void*) new char[size_in_bytes]);
#endif
}

// Unmap device memory when done
void deallocate_mem(cl_object &cl_obj, void *ptr, uint64_t idx) {
#ifdef __VITIS_CL__
    cl_int err;

    cl::Buffer *buffer = &cl_obj.buffers[idx];

    OCL_CHECK(err, err = cl_obj.q.enqueueUnmapMemObject(*buffer, ptr));
    OCL_CHECK(err, err = cl_obj.q.finish());
#else
    delete[] (char *) ptr;
#endif
}
