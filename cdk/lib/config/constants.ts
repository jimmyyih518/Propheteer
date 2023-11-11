export interface PropheteerStackProps {
  name_prefix: string;
  precompute_worker_image: string;
}

export const propheteerStackProps: PropheteerStackProps = {
  name_prefix: "Propheteer",
  precompute_worker_image:
    "185731724227.dkr.ecr.us-west-2.amazonaws.com/propheteer-precompute-worker-ecr:latest",
};
