#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <string>

#include "vm.hpp"

namespace {

std::string writeTempPTXFile(const std::string& contents)
{
    std::array<char, L_tmpnam> pathBuffer {};
    char* rawPath = std::tmpnam(pathBuffer.data());
    EXPECT_NE(rawPath, nullptr);

    std::string path = rawPath ? rawPath : "";
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

float runSingleElementVectorAdd(const std::string& ptxPath, float a, float b)
{
    PTXVM vm;
    EXPECT_TRUE(vm.initialize());
    EXPECT_TRUE(vm.loadProgram(ptxPath));

    const CUdeviceptr inputAPtr = vm.allocateMemory(sizeof(float));
    const CUdeviceptr inputBPtr = vm.allocateMemory(sizeof(float));
    const CUdeviceptr outputPtr = vm.allocateMemory(sizeof(float));
    EXPECT_NE(inputAPtr, 0u);
    EXPECT_NE(inputBPtr, 0u);
    EXPECT_NE(outputPtr, 0u);

    float output = 0.0f;
    EXPECT_TRUE(vm.copyMemoryHtoD(inputAPtr, &a, sizeof(float)));
    EXPECT_TRUE(vm.copyMemoryHtoD(inputBPtr, &b, sizeof(float)));

    std::vector<KernelParameter> params;
    params.push_back({inputAPtr, sizeof(uint64_t), 0});
    params.push_back({inputBPtr, sizeof(uint64_t), 8});
    params.push_back({outputPtr, sizeof(uint64_t), 16});
    params.push_back({static_cast<CUdeviceptr>(1), sizeof(uint32_t), 24});

    vm.setKernelParameters(params);
    vm.getExecutor().setGridDimensions(1, 1, 1, 32, 1, 1);

    EXPECT_TRUE(vm.run());
    EXPECT_TRUE(vm.copyMemoryDtoH(&output, outputPtr, sizeof(float)));

    vm.freeMemory(inputAPtr);
    vm.freeMemory(inputBPtr);
    vm.freeMemory(outputPtr);
    return output;
}

}  // namespace

TEST(BarraCUDAPTXCompatibilityTest, RunsVectorAddWithSetpLtU32)
{
    const std::string ptx = R"(
.version 8.0
.target sm_89
.address_size 64

.entry vector_add (
    .param .u64 param0,
    .param .u64 param1,
    .param .u64 param2,
    .param .u32 param3
)
{
    .reg .u32  %r<7>;
    .reg .u64  %rd<10>;
    .reg .f32  %f<4>;
    .reg .pred %p<2>;

$BB0:
    ld.param.u64 %rd1, [param0];
    ld.param.u64 %rd2, [param1];
    ld.param.u64 %rd3, [param2];
    ld.param.u32 %r1, [param3];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mul.lo.u32 %r5, %r3, %r4;
    add.u32 %r6, %r2, %r5;
    setp.lt.u32 %p1, %r6, %r1;
    @%p1 bra $BB1;
    bra $BB2;
$BB1:
    cvt.u64.u32 %rd5, %r6;
    mad.lo.u64 %rd4, %rd5, 4, %rd3;
    cvt.u64.u32 %rd7, %r6;
    mad.lo.u64 %rd6, %rd7, 4, %rd1;
    ld.global.f32 %f1, [%rd6];
    cvt.u64.u32 %rd9, %r6;
    mad.lo.u64 %rd8, %rd9, 4, %rd2;
    ld.global.f32 %f2, [%rd8];
    add.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd4], %f3;
    bra $BB2;
$BB2:
    exit;
}
)";

    const std::string ptxPath = writeTempPTXFile(ptx);
    ASSERT_FALSE(ptxPath.empty());

    constexpr std::array<float, 4> inputA = {1.0f, 2.0f, 3.0f, 4.0f};
    constexpr std::array<float, 4> inputB = {5.0f, 6.0f, 7.0f, 8.0f};
    std::array<float, 4> output = {};

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = runSingleElementVectorAdd(ptxPath, inputA[i], inputB[i]);
    }

    EXPECT_FLOAT_EQ(output[0], 6.0f);
    EXPECT_FLOAT_EQ(output[1], 8.0f);
    EXPECT_FLOAT_EQ(output[2], 10.0f);
    EXPECT_FLOAT_EQ(output[3], 12.0f);
    std::remove(ptxPath.c_str());
}
