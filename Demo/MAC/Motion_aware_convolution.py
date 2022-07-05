import cupy
import torch
import numpy
import re
import math

MAC_updateOutput = '''
extern "C" __global__ void MAC_updateOutput(
    const int n,
    const int DILATION,
    const float* input,
    const float* Kernel_H,
    const float* Kernel_W,
    const float* Offset_X,
    const float* Offset_Y,
    float* output
    ){
        for(int intIndex=blockDim.x*blockIdx.x+threadIdx.x; intIndex<n ; intIndex+=gridDim.x*blockDim.x)
        {
            int w = intIndex % SIZE_3(output);
            int h = (intIndex / SIZE_3(output)) % SIZE_2(output);
            int c = (intIndex / SIZE_3(output) / SIZE_2(output)) % SIZE_1(output);
            int b = (intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output)) % SIZE_0(output);
            int FilterSize = SIZE_1(Kernel_H) / SIZE_1(Offset_X);
            float tmpOut=0.0;
            for(int count=0; count<SIZE_1(Offset_X); count++)
            {
                float offset_x = VALUE_4(Offset_X,b,count,h,w);
                float offset_y = VALUE_4(Offset_Y,b,count,h,w);
                int dx = (int)offset_x;
                int dy = (int)offset_y;
                int sign_x = 0;
                int sign_y = 0;
                float alpha = offset_x-(float)dx;
                float beta = offset_y-(float)dy;  
                
                if(offset_x > 1)
                    sign_x = 1;
                if(offset_x < 1)
                    sign_x = -1;
                if(offset_y > 1)
                    sign_y = 1;
                if(offset_y < 1)
                    sign_y = -1;
                    
                int Left_X = w+sign_x*count*DILATION+dx;
                if(Left_X < 0)
                    Left_X = 0;
                if(Left_X > SIZE_3(output)-1)
                    Left_X = SIZE_3(output)-1;
                        
                int Right_X = w+sign_x*count*DILATION+dx+1;
                if(Right_X < 0)
                    Right_X = 0;
                if(Right_X > SIZE_3(output)-1)
                    Right_X = SIZE_3(output)-1;
                        
                int Top_Y = h+sign_y*count*DILATION+dy;
                if(Top_Y < 0)
                    Top_Y = 0;
                if(Top_Y > SIZE_2(output)-1)
                    Top_Y = SIZE_2(output)-1;
                        
                int Bottom_Y = h+sign_y*count*DILATION+dy+1;
                if(Bottom_Y < 0)
                    Bottom_Y = 0;
                if(Bottom_Y > SIZE_2(output)-1)
                    Bottom_Y = SIZE_2(output)-1;
                
                for(int KH=0; KH<FilterSize; KH++)
                {
                    for(int KW=0; KW<FilterSize; KW++)
                    {
                        tmpOut += VALUE_4(Kernel_H,b,count*FilterSize+KH,h,w)*VALUE_4(Kernel_W,b,count*FilterSize+KW,h,w)
                                *(
                                    (1-alpha)*(1-beta)*VALUE_4(input,b,c,Top_Y+KH,Left_X+KW)+
                                    (1-beta)*alpha*VALUE_4(input,b,c,Top_Y+KH,Right_X+KW)+
                                    beta*(1-alpha)*VALUE_4(input,b,c,Bottom_Y+KH,Left_X+KW)+
                                    beta*alpha*VALUE_4(input,b,c,Bottom_Y+KH,Right_X+KW)
                                );
                    }                
                }
            }
            output[intIndex] = tmpOut;
        }
}
'''
MAC_updateGradKernel = '''
extern "C" __global__ void MAC_updateGradKernel(
    const int n,
    const int DILATION,
    const float* input,
    const float* gradOut,
    const float* KernelH,
    const float* KernelW,
    const float* OffsetX,
    const float* OffsetY,
    float* GradKernelH,
    float* GradKernelW
){
    for(int intIndex = (blockDim.x*blockIdx.x)+threadIdx.x; intIndex<n; intIndex+=gridDim.x*blockDim.x)
    {
        int w = intIndex % SIZE_3(GradKernelH);
        int h = (intIndex / SIZE_3(GradKernelH)) % SIZE_2(GradKernelH);
        int c = (intIndex / SIZE_3(GradKernelH) / SIZE_2(GradKernelH)) % SIZE_1(GradKernelH);
        int b = (intIndex / SIZE_3(GradKernelH) / SIZE_2(GradKernelH) / SIZE_1(GradKernelH)) % SIZE_0(GradKernelH);
        float tempOutH = 0.0;
        float tempOutW = 0.0;
        int FilterSize = SIZE_1(KernelH) / SIZE_1(OffsetY);
        int PART = c / FilterSize;
        int Position = c % FilterSize;
        int channels = SIZE_1(gradOut);
        float offset_x = VALUE_4(OffsetX,b,PART,h,w);
        float offset_y = VALUE_4(OffsetY,b,PART,h,w);
        int dx = (int)offset_x;
        int dy = (int)offset_y;
        int sign_x = 0;
        int sign_y = 0;
        float alpha = offset_x - (float)dx;
        float beta = offset_y - (float)dy;
                       
        if(offset_x > 1)
            sign_x = 1;
        if(offset_x < 1)
            sign_x = -1;
        if(offset_y > 1)
            sign_y = 1;
        if(offset_y < 1)
            sign_y = -1;
            
        int Left_X = w+sign_x*PART*DILATION+dx;
        if(Left_X < 0)
            Left_X = 0;
        if(Left_X > SIZE_3(GradKernelH)-1)
            Left_X = SIZE_3(GradKernelH) - 1;
            
        int Right_X = w+sign_x*PART*DILATION+dx+1;
        if(Right_X < 0)
            Right_X = 0;
        if(Right_X > SIZE_3(GradKernelH)-1)
            Right_X = SIZE_3(GradKernelH) - 1;
        
        int Top_Y = h+sign_y*PART*DILATION+dy;
        if(Top_Y < 0)
            Top_Y = 0;
        if(Top_Y > SIZE_2(GradKernelH)-1)
            Top_Y = SIZE_2(GradKernelH) - 1;
        
        int Bottom_Y = h+sign_y*PART*DILATION+dy+1;
        if(Bottom_Y < 0)
            Bottom_Y = 0;
        if(Bottom_Y > SIZE_2(GradKernelH)-1)
            Bottom_Y = SIZE_2(GradKernelH) - 1;
        
        for(int channel = 0; channel<channels; channel++)
        {
            for(int K_xy = 0; K_xy < FilterSize; K_xy++)
            {
                tempOutH += VALUE_4(gradOut,b,channel,h,w)*VALUE_4(KernelW,b,PART*FilterSize+K_xy,h,w)*(
                    (1-beta)*(1-alpha)*VALUE_4(input,b,channel,Top_Y+Position,Left_X+K_xy)+
                    (1-beta)*alpha*VALUE_4(input,b,channel,Top_Y+Position,Right_X+K_xy)+
                    beta*(1-alpha)*VALUE_4(input,b,channel,Bottom_Y+Position,Left_X+K_xy)+
                    beta*alpha*VALUE_4(input,b,channel,Bottom_Y+Position,Right_X+K_xy)
                );
                
                tempOutW += VALUE_4(gradOut,b,channel,h,w)*VALUE_4(KernelH,b,PART*FilterSize+K_xy,h,w)*(
                    (1-beta)*(1-alpha)*VALUE_4(input,b,channel,Top_Y+K_xy,Left_X+Position)+
                    (1-beta)*alpha*VALUE_4(input,b,channel,Top_Y+K_xy,Right_X+Position)+
                    beta*(1-alpha)*VALUE_4(input,b,channel,Bottom_Y+K_xy,Left_X+Position)+
                    beta*alpha*VALUE_4(input,b,channel,Bottom_Y+K_xy,Right_X+Position)
                );
            }
        }
        GradKernelH[intIndex] = tempOutH;
        GradKernelW[intIndex] = tempOutW;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    }
}
'''

MAC_updateGradOffsetX = '''
extern "C" __global__ void MAC_updateGradOffsetX(
    const int n,
    const int DILATION,
    const float* gradOut,
    const float* input,
    const float* Kernel_H,
    const float* Kernel_W,
    const float* Offset_X,
    const float* Offset_Y,
    float* GradOffsetX
    ){
    for(int intIndex=(blockDim.x*blockIdx.x)+threadIdx.x ; intIndex<n; intIndex += gridDim.x * blockDim.x)
    {
        int w = intIndex % SIZE_3(GradOffsetX);
        int h = (intIndex / SIZE_3(GradOffsetX)) % SIZE_2(GradOffsetX);
        int c = (intIndex / SIZE_3(GradOffsetX) / SIZE_2(GradOffsetX)) % SIZE_1(GradOffsetX);
        int b = (intIndex / SIZE_3(GradOffsetX) / SIZE_2(GradOffsetX) / SIZE_1(GradOffsetX)) % SIZE_0(GradOffsetX);
        
        int FilterSize = SIZE_1(Kernel_H) / SIZE_1(Offset_X);
        int Channels = SIZE_1(gradOut);
        float tmpGrad = 0.0;
        float offset_x = VALUE_4(Offset_X,b,c,h,w);
        float offset_y = VALUE_4(Offset_Y,b,c,h,w);
        int sign_x = 0;
        int sign_y = 0;
        int dx = (int)offset_x;
        int dy = (int)offset_y;
        float beta = offset_y - (float)dy;
        
        if(offset_x > 1)
            sign_x = 1;
        if(offset_x < 1)
            sign_x = -1;
        if(offset_y > 1)
            sign_y = 1;
        if(offset_y < 1)
            sign_y = -1;
            
        int Left_X = w+sign_x*c*DILATION+dx;
        if(Left_X < 0)
            Left_X = 0;
        if(Left_X > SIZE_3(GradOffsetX)-1)
            Left_X = SIZE_3(GradOffsetX)-1;
        
        int Right_X = w+sign_x*c*DILATION+dx+1;
        if(Right_X < 0)
            Right_X = 0;
        if(Right_X > SIZE_3(GradOffsetX)-1)
            Right_X = SIZE_3(GradOffsetX)-1;
        
        int Top_Y = h+sign_y*c*DILATION+dy;
        if(Top_Y < 0)
            Top_Y = 0;
        if(Top_Y > SIZE_2(GradOffsetX)-1)
            Top_Y = SIZE_2(GradOffsetX)-1;
        
        int Bottom_Y = h+sign_y*c*DILATION+dy+1;
        if(Bottom_Y < 0)
            Bottom_Y = 0;
        if(Bottom_Y > SIZE_2(GradOffsetX)-1)
            Bottom_Y = SIZE_2(GradOffsetX)-1;
        
        for(int channel=0; channel<Channels ; channel++)
        {
            for(int KH=0; KH<FilterSize ; KH++)
            {
                for(int KW=0; KW<FilterSize; KW++)
                {
                    tmpGrad += VALUE_4(gradOut,b,channel,h,w)*VALUE_4(Kernel_H,b,c*FilterSize+KH,h,w)
                            *VALUE_4(Kernel_W,b,c*FilterSize+KW,h,w)
                            *(
                                -(1-beta)*VALUE_4(input,b,channel,Top_Y+KH,Left_X+KW)+
                                (1-beta)*VALUE_4(input,b,channel,Top_Y+KH,Right_X+KW)-
                                beta*VALUE_4(input,b,channel,Bottom_Y+KH,Left_X+KW)+
                                beta*VALUE_4(input,b,channel,Bottom_Y+KH,Right_X+KW)
                            );
                }
            }
        }
        GradOffsetX[intIndex] = tmpGrad;
    }
}
'''

MAC_updateGradOffsetY = '''
extern "C" __global__ void MAC_updateGradOffsetY(
    const int n,
    const int DILATION,
    const float* gradOut,
    const float* input,
    const float* Kernel_H,
    const float* Kernel_W,
    const float* Offset_X,
    const float* Offset_Y,
    float* GradOffsetY
    ){
        for(int intIndex=blockDim.x*blockIdx.x+threadIdx.x; intIndex<n; intIndex+=gridDim.x*blockDim.x)
        {
            int w = intIndex % SIZE_3(GradOffsetY);
            int h = (intIndex / SIZE_3(GradOffsetY)) % SIZE_2(GradOffsetY);
            int c = (intIndex / SIZE_3(GradOffsetY) / SIZE_2(GradOffsetY)) % SIZE_1(GradOffsetY);
            int b = (intIndex / SIZE_3(GradOffsetY) / SIZE_2(GradOffsetY) / SIZE_1(GradOffsetY)) % SIZE_0(GradOffsetY);
            
            const int FilterSize = SIZE_1(Kernel_H) / SIZE_1(Offset_Y);
            const int Channels = SIZE_1(gradOut);
            float tmpGrad = 0.0;
            float offset_x = VALUE_4(Offset_X,b,c,h,w);
            float offset_y = VALUE_4(Offset_Y,b,c,h,w);
            int sign_x = 0;
            int sign_y = 0;
            int dx = (int)offset_x;
            int dy = (int)offset_y;
            float alpha = offset_x - (float)dx;
            
            if(offset_x > 1)
                sign_x = 1;
            if(offset_x < 1)
                sign_x = -1;
            if(offset_y > 1)
                sign_y = 1;
            if(offset_y < 1)
                sign_y = -1;
                
            int Left_X = w+sign_x*c*DILATION+dx;
            if(Left_X < 0)
                Left_X = 0;
            if(Left_X > SIZE_3(GradOffsetY)-1)
                Left_X = SIZE_3(GradOffsetY)-1;
            
            int Right_X = w+sign_x*c*DILATION+dx+1;
            if(Right_X < 0)
                Right_X = 0;
            if(Right_X > SIZE_3(GradOffsetY)-1)
                Right_X = SIZE_3(GradOffsetY)-1;
            
            int Top_Y = h+sign_y*c*DILATION+dy;
            if(Top_Y < 0)
                Top_Y = 0;
            if(Top_Y > SIZE_2(GradOffsetY)-1)
                Top_Y = SIZE_2(GradOffsetY)-1;
            
            int Bottom_Y = h+sign_y*c*DILATION+dy+1;
            if(Bottom_Y < 0)
                Bottom_Y = 0;
            if(Bottom_Y > SIZE_2(GradOffsetY)-1)
                Bottom_Y = SIZE_2(GradOffsetY)-1;
            
            for(int channel=0; channel<Channels; channel++)
            {
                for(int KH=0; KH<FilterSize; KH++)
                {
                    for(int KW=0; KW<FilterSize; KW++)
                    {
                        tmpGrad += VALUE_4(gradOut,b,channel,h,w)*VALUE_4(Kernel_H,b,c*FilterSize+KH,h,w)
                                *VALUE_4(Kernel_W,b,c*FilterSize+KW,h,w)
                                *(
                                    -(1-alpha)*VALUE_4(input,b,channel,Top_Y+KH,Left_X+KW)+
                                    (1-alpha)*VALUE_4(input,b,channel,Bottom_Y+KH,Left_X+KW)-
                                    alpha*VALUE_4(input,b,channel,Top_Y+KH,Right_X+KW)+
                                    alpha*VALUE_4(input,b,channel,Bottom_Y+KH,Right_X+KW)
                                );
                    }
                }
            }
            GradOffsetY[intIndex] = tmpGrad;
        }
}
'''

def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break

        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))

    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))',strKernel)
        if objMatch is None:
            break

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0),strTensor+'['+str.join('+',strIndex)+']')

    return strKernel

@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.RawKernel(strKernel,strFunction)

class _FunctionMultiVK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,kernelH,kernelW,offsetX,offsetY,dilation):
        ctx.save_for_backward(input,kernelH,kernelW,offsetX,offsetY)
        ctx.dilation = dilation
        intBatchSize = input.size(0)
        intChannel = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = int(kernelH.size(1)//offsetX.size(1))

        intOutputHeight = kernelH.size(2)
        intOutputWidth = kernelH.size(3)

        assert(intInputHeight-((intFilterSize-1)*dilation+1)+1 == intOutputHeight)
        assert(intInputWidth-((intFilterSize-1)*dilation+1)+1== intOutputWidth)

        assert(input.is_contiguous() == True)
        assert(kernelH.is_contiguous() == True)
        assert(kernelW.is_contiguous() == True)
        assert(offsetX.is_contiguous() == True)
        assert(offsetY.is_contiguous() == True)

        output = input.new_zeros(intBatchSize,intChannel,intOutputHeight,intOutputWidth)

        if input.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream
            n = output.nelement()
            cupy_launch('MAC_updateOutput',cupy_kernel('MAC_updateOutput',{
                'input':input,
                'Kernel_H':kernelH,
                'Kernel_W':kernelW,
                'Offset_X':offsetX,
                'Offset_Y':offsetY,
                'output':output
            }))(
                grid = tuple([int((n+512-1)/512),1,1]),
                block = tuple([512,1,1]),
                args = [cupy.int32(n),cupy.int32(dilation),input.data_ptr(),kernelH.data_ptr(),kernelW.data_ptr(),offsetX.data_ptr(),offsetY.data_ptr(),output.data_ptr()],
                stream = Stream
            )
        elif input.is_cuda == False:
            raise NotImplementedError()

        return output

    @staticmethod
    def backward(ctx,gradOutput):
        input,kernelH,kernelW,offsetX,offsetY = ctx.saved_tensors
        dilation = ctx.dilation
        intBatchSize = input.size(0)
        intChannel = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = int(kernelH.size(1)//offsetX.size(1))

        intOutputHeight = kernelH.size(2)
        intOutputWidth = kernelH.size(3)

        assert (intInputHeight - ((intFilterSize - 1) * dilation + 1) + 1 == intOutputHeight)
        assert (intInputWidth - ((intFilterSize - 1) * dilation + 1) + 1 == intOutputWidth)

        gradOutput = gradOutput.contiguous(); assert (gradOutput.is_cuda==True)

        gradInput = input.new_zeros(intBatchSize,intChannel,intInputHeight,intInputWidth)
        gradKernelH = input.new_zeros(intBatchSize,kernelH.size(1),intOutputHeight,intOutputWidth)
        gradKernelW = input.new_zeros(intBatchSize,kernelW.size(1),intOutputHeight,intOutputWidth)
        gradoffsetX = input.new_zeros(intBatchSize,offsetX.size(1),intOutputHeight,intOutputWidth)
        gradoffsetY = input.new_zeros(intBatchSize,offsetY.size(1),intOutputHeight,intOutputWidth)
        if input.is_cuda == True:

            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream
            #update Kernel

            n_Kernel = gradKernelH.nelement()
            cupy_launch('MAC_updateGradKernel',cupy_kernel('MAC_updateGradKernel',{
                'input': input,
                'gradOut': gradOutput,
                'KernelH':kernelH,
                'KernelW':kernelW,
                'OffsetX':offsetX,
                'OffsetY':offsetY,
                'GradKernelH':gradKernelH,
                'GradKernelW':gradKernelW
            }))(
                grid = tuple([int((n_Kernel+512-1)/512),1,1]),
                block = tuple([512,1,1]),
                args=[cupy.int32(n_Kernel),cupy.int32(dilation),input.data_ptr(),gradOutput.data_ptr(),kernelH.data_ptr(),kernelW.data_ptr()
                        ,offsetX.data_ptr(),offsetY.data_ptr(),gradKernelH.data_ptr(),gradKernelW.data_ptr()],
                stream = Stream
            )
            n_offsetX = gradoffsetX.nelement()
            cupy_launch('MAC_updateGradOffsetX',cupy_kernel('MAC_updateGradOffsetX',{
                'gradOut':gradOutput,
                'input':input,
                'Kernel_H':kernelH,
                'Kernel_W':kernelW,
                'Offset_X':offsetX,
                'Offset_Y':offsetY,
                'GradOffsetX':gradoffsetX
            }))(
                grid = tuple([int((n_offsetX+512-1)/512),1,1]),
                block = tuple([512,1,1]),
                args=[cupy.int32(n_offsetX),cupy.int32(dilation),gradOutput.data_ptr(),input.data_ptr(),kernelH.data_ptr(),kernelW.data_ptr(),offsetX.data_ptr(),offsetY.data_ptr(),gradoffsetX.data_ptr()],
                stream = Stream
            )

            #update offsetY
            n_offsetY = gradoffsetY.nelement()
            cupy_launch('MAC_updateGradOffsetY',cupy_kernel('MAC_updateGradOffsetY',{
                'gradOut':gradOutput,
                'input':input,
                'Kernel_H':kernelH,
                'Kernel_W':kernelW,
                'Offset_X':offsetX,
                'Offset_Y':offsetY,
                'GradOffsetY':gradoffsetY
            }))(
                grid = tuple([int((n_offsetY+512-1)/512),1,1]),
                block = tuple([512,1,1]),
                args=[cupy.int32(n_offsetY),cupy.int32(dilation),gradOutput.data_ptr(),input.data_ptr(),kernelH.data_ptr(),kernelW.data_ptr(),offsetX.data_ptr(),offsetY.data_ptr(),gradoffsetY.data_ptr()],
                stream = Stream
            )

        #end
        elif input.is_cuda == False:

            raise NotImplementedError()

        return gradInput,gradKernelH,gradKernelW,gradoffsetX,gradoffsetY,None

#end
