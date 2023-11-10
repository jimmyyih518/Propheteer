import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface S3BucketProps {
  bucketName: string;
  bucketAccessRoleArn: string;
}

export class CreateS3Bucket extends Construct {
  public readonly bucket: s3.Bucket;
  public readonly role: iam.IRole;

  constructor(scope: Construct, id: string, props: S3BucketProps) {
    super(scope, id);

    // Create the S3 bucket
    this.bucket = new s3.Bucket(this, props.bucketName, {
      bucketName: props.bucketName,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Create the IAM role
    this.role = iam.Role.fromRoleArn(
      this,
      `${props.bucketName}AccessRole`,
      props.bucketAccessRoleArn,
      {
        // Setting 'mutable' to false as the role is imported and not created within this stack
        mutable: false,
      },
    );

    // Grant the role full access to the S3 bucket
    this.bucket.grantReadWrite(this.role);
  }
}
