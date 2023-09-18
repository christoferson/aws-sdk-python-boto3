
def run_demo(session):

    # Let's use Amazon S3
    s3 = session.resource('s3')

    # Print out bucket names
    for bucket in s3.buckets.all():
        print(bucket.name)
