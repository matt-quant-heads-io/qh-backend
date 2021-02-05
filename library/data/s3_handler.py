import boto3


class s3Handler(object):
    def __init__(self, resource='s3',
                 aws_access_key_id='AKIAIWPOHHSFY2F4UEOA',
                 aws_secret_access_key='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'):
        self._conn = self.__create_resource(resource, aws_access_key_id, aws_secret_access_key)

    def __create_resource(self, resource=None, aws_access_key_id=None, aws_secret_access_key=None):
        try:
            if resource is None or aws_access_key_id is None or aws_secret_access_key is None:
                msg = "s3Handler parameters cannot be None type"
                raise Exception(msg)
            else:
                conn = boto3.client(
                    resource,
                    aws_access_key_id,
                    aws_secret_access_key
                )
                return conn
        except Exception as e:
            msg = "Error in __create_resource. Check params for boto client: check resource, aws_access_key_id," \
                  "aws_secret_access_key"
            raise e(msg)






conn = boto3.client(
    's3',

)