from ..op_builder import CommBuilder

common_cpp_module = None


def build_op():
    global common_cpp_module
    builder = CommBuilder()
    try:
        common_cpp_module = builder.load()
        print(f'MCR-DL {builder.absolute_name()} built successfully')
        return common_cpp_module
    except Exception as inst:
        # if comm cannot be built, use torch.dist.
        print(f"Failed to build {builder.absolute_name()}. Full error: {inst}")
        exit(0)