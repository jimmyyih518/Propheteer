import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";
import { CreateS3Bucket } from "./constructs/s3_bucket_create";
import { propheteerStackProps } from "./config/constants";

export class PropheteerApiStack extends cdk.Stack {
  constructor(
    scope: Construct,
    id: string,
    props: cdk.StackProps & typeof propheteerStackProps,
  ) {
    super(scope, id, props);

    // Main workflow role
    const operatorRole = new iam.Role(
      this,
      `${props.name_prefix}OperatorRole`,
      {
        assumedBy: new iam.ServicePrincipal("*"),
      },
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
  }
}
