
from prefect import serve
from pipeline_prefect import all_flow, train_flow, evaluate_flow

if __name__ == "__main__":
    all_deployment = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 2 * * *"
    )

    train_deployment = train_flow.to_deployment(
        name="ml-pipeline-train"
    )

    evaluate_deployment = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate"
    )

    serve(
        all_deployment,
        train_deployment,
        evaluate_deployment
    )
