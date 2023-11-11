import * as cdk from "aws-cdk-lib";
import * as logs from "aws-cdk-lib/aws-logs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecs from "aws-cdk-lib/aws-ecs";
import { Construct } from "constructs";

export interface EcsProps {
  name: string;
  taskRoleArn: string;
  dockerImageArn: string;
}

export class CreateEcsService extends Construct {
  public readonly ecsArn: string;
  public readonly ecsService: ecs.FargateService;

  constructor(scope: Construct, id: string, props: EcsProps) {
    super(scope, id);

    // Import operator role to grant access permissions
    const taskRole = iam.Role.fromRoleArn(
      this,
      `${props.name}ImportedTaskRole`,
      props.taskRoleArn,
    );
    // Create a Log Group
    const serviceLogGroup = new logs.LogGroup(this, "MyLogGroup", {
      logGroupName: `/ecs/${props.name}EcsServiceLogs`,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ECS cluster for running the docker image
    const EcsCluster = new ecs.Cluster(this, `${props.name}EcsCluster`, {});
    // Define the task definition with the existing ECR image
    const TaskDefinition = new ecs.FargateTaskDefinition(
      this,
      `${props.name}EcsTaskDef`,
      {
        executionRole: taskRole,
      },
    );
    TaskDefinition.addContainer(`${props.name}Container`, {
      image: ecs.ContainerImage.fromRegistry(props.dockerImageArn),
      memoryLimitMiB: 512,
      cpu: 256,
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: `${props.name}Service`,
        logGroup: serviceLogGroup,
      }),
    });

    // Fargate service to manage the ECS worker
    this.ecsService = new ecs.FargateService(this, `${props.name}EcsService`, {
      cluster: EcsCluster,
      taskDefinition: TaskDefinition,
    });

    this.ecsArn = this.ecsService.serviceArn;
  }
}
