import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";
import { CreateS3Bucket } from "./constructs/s3_bucket_create";
import { CreateCompositeRole } from "./constructs/iam_composite_role";
import { PropheteerStackProps } from "./config/constants";

export class PropheteerApiStack extends cdk.Stack {
  constructor(
    scope: Construct,
    id: string,
    props: cdk.StackProps & PropheteerStackProps,
  ) {
    super(scope, id, props);

    // Main workflow role
    const operatorRole = new CreateCompositeRole(
      this,
      `${props.name_prefix}OperatorRole`,
      { name: `${props.name_prefix}OperatorRole` },
    );

    // Model Artifacts Bucket
    const modelArtifactsBucket = new CreateS3Bucket(
      this,
      `${props.name_prefix}ModelArtifacts`,
      {
        bucketName: `${props.name_prefix.toLowerCase()}-model-artifacts`,
        bucketAccessRoleArn: operatorRole.roleArn,
      },
    );

    // Export Role Arn Export
    const operatorRoleArn = new cdk.CfnOutput(
      this,
      `${props.name_prefix}OperatorRoleArn`,
      {
        value: operatorRole.roleArn,
        exportName: `${props.name_prefix}-OperatorRoleArn`,
      },
    );
  }
}
