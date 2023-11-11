import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";
import { PropheteerStackProps } from "./config/constants";

export class PropheteerPrecomputeStack extends cdk.Stack {
  constructor(
    scope: Construct,
    id: string,
    props: cdk.StackProps & PropheteerStackProps,
  ) {
    super(scope, id, props);

    // Import operator role to grant access permissions
    const operatorRoleArn = cdk.Fn.importValue(
      `${props.name_prefix}-OperatorRoleArn`,
    );
    const operatorRole = iam.Role.fromRoleArn(
      this,
      "ImportedOperatorRole",
      operatorRoleArn,
    );

    // ECR Repo for Docker Images used by ECS workers
    const precomputeWorkerEcr = new ecr.Repository(
      this,
      `${props.name_prefix}PrecomputeWorkerEcr`,
      {
        repositoryName: `${props.name_prefix.toLowerCase()}-precompute-worker-ecr`,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      },
    );
    precomputeWorkerEcr.grantPullPush(operatorRole);
  }
}
