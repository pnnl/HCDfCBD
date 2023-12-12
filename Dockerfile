FROM public.ecr.aws/lambda/python:3.11

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install --default-timeout=100 -r requirements.txt

# Copy function code
COPY lambda/lambda_function.py ${LAMBDA_TASK_ROOT}

# Make the diomics library available
COPY diomics ${LAMBDA_TASK_ROOT}/diomics

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.handler" ]