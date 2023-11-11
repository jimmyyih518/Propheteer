#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { PropheteerApiStack } from "../lib/propheteer_api_stack";
import { PropheteerPrecomputeStack } from "../lib/propheteer_precompute_worker_stack";
import { propheteerStackProps } from "../lib/config/constants";

const app = new cdk.App();

// CFN Stacks to be deployed
const stackDefinitions = [
  { constructor: PropheteerApiStack, name: "PropheteerApiStack" },
  { constructor: PropheteerPrecomputeStack, name: "PropheteerPrecomputeStack" },
];


// Sequential deployments to handle dependencies (cfn outputs)
let previousStack: cdk.Stack | null = null;

stackDefinitions.forEach((stackDef) => {
  const stack = new stackDef.constructor(app, stackDef.name, propheteerStackProps);

  if (previousStack) {
    stack.addDependency(previousStack);
  }

  previousStack = stack;
});
