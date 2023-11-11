import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface RoleProps {
  name: string;
}

export class CreateCompositeRole extends Construct {
  public readonly role: iam.IRole;
  public readonly roleArn: string;

  constructor(scope: Construct, id: string, props: RoleProps) {
    super(scope, id);

    this.role = new iam.Role(this, `${props.name}`, {
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("ec2.amazonaws.com"),
        new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        new iam.ServicePrincipal("states.amazonaws.com"),
        new iam.ServicePrincipal("lambda.amazonaws.com"),
      ),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AdministratorAccess",
        ),
      ],
    });

    this.roleArn = this.role.roleArn;
  }
}
