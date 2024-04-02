import boto3
from sagemaker.estimator import Estimator
from sagemaker import get_execution_role

account_id = boto3.client('sts').get_caller_identity().get('Account')

ecr_repository = 'holiday-check'
tag = ':latest'

region = boto3.session.Session().region_name

uri_suffix = 'amazonaws.com'
if region in ['cn-north-1', 'cn-northwest-1']:
    uri_suffix = 'amazonaws.com.cn'

byoc_image_uri = \
    '{}.dkr.ecr.{}.{}/{}'.format(
        account_id,
        region,
        uri_suffix,
        ecr_repository + tag
    )

print(byoc_image_uri)

hyper_parameters = {
    "bucket": "022-holiday-check",
    "training_specification_xml":
    "training_specification/training_specification.xml"
}
instance_type = 'local'  # Replace with 'local' if needed.
estimator = Estimator(
    image_uri=byoc_image_uri,
    hyperparameters=hyper_parameters,
    role=get_execution_role(),
    instance_count=1,
    instance_type=instance_type
)

estimator.fit()

