//! (No HS analog.) The RL API (`/api/v1`, feature `rl`) on Tutorial.spthy:
//! the routes' bodies and errors, a proof walked through `execute` and
//! checked with `check-proof`, and parallel requests giving the same answers
//! as sequential ones.

#![cfg(feature = "rl")]

mod common;

use std::collections::BTreeMap;

use serde_json::{json, Value};

use common::{maude_available, start_server_with_theory, TestServer};

fn lemma_url(s: &TestServer, lemma: &str, route: &str) -> String {
    s.url(&format!("/api/v1/theories/1/lemmas/{lemma}/{route}"))
}

async fn get_json(s: &TestServer, url: &str) -> (u16, Value) {
    let res = s.client.get(url).send().await.expect("send");
    let status = res.status().as_u16();
    (status, res.json().await.expect("json body"))
}

async fn post_form(s: &TestServer, url: &str, form: &[(&str, &str)]) -> (u16, String) {
    let res = s.client.post(url).form(form).send().await.expect("send");
    let status = res.status().as_u16();
    (status, res.text().await.expect("body"))
}

async fn initial(s: &TestServer, lemma: &str) -> Value {
    let (status, body) = get_json(s, &lemma_url(s, lemma, "initial-system")).await;
    assert_eq!(status, 200, "{body}");
    body
}

async fn execute(s: &TestServer, lemma: &str, sys: &Value, i: usize) -> Value {
    let form = [
        ("proofmethod", sys["encodedMethods"][i].as_str().unwrap()),
        ("system", sys["encodedSys"].as_str().unwrap()),
    ];
    let (status, body) = post_form(s, &lemma_url(s, lemma, "execute"), &form).await;
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).unwrap()
}

fn errors(body: &str) -> Value {
    serde_json::from_str::<Value>(body).unwrap()["errors"].clone()
}

#[tokio::test(flavor = "multi_thread")]
async fn theories_and_the_start_of_a_lemma() {
    if !maude_available() {
        return;
    }
    let s = start_server_with_theory("Tutorial.spthy").await;
    let (status, body) = get_json(&s, &s.url("/api/v1/theories")).await;
    assert_eq!(status, 200);
    assert_eq!(
        body,
        json!({"Tutorial": {"theoryIndex": 1, "theoryName": "Tutorial", "theoryKind": "trace"}})
    );

    let sys = initial(&s, "Client_auth").await;
    assert_eq!(sys["status"], "Undertermined");
    let methods = sys["proofMethods"].as_array().unwrap();
    assert_eq!(
        methods.len(),
        sys["encodedMethods"].as_array().unwrap().len()
    );
    let names: Vec<&str> = methods
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert!(
        names.contains(&"Simplify") && names.contains(&"Induction"),
        "{names:?}"
    );
    assert!(sys["encodedSys"].as_str().unwrap().len() > 10);

    // Simplify on the start system gives one case, with goals in the fork's
    // format.
    let simplify = names.iter().position(|n| *n == "Simplify").unwrap();
    let cases = execute(&s, "Client_auth", &sys, simplify).await;
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 1, "{cases:?}");
    let goal = cases[0]["system"]["proofMethods"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == "SolveGoal")
        .expect("a goal after simplify")
        .clone();
    let kind = goal["goal"]["type"].as_str().unwrap();
    assert!(
        ["Action", "Chain", "Premise", "Split", "Disj", "Subterm"].contains(&kind),
        "{goal}"
    );
    assert!(goal["goal"]["args"].as_array().is_some(), "{goal}");
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_follow_the_fork() {
    if !maude_available() {
        return;
    }
    let s = start_server_with_theory("Tutorial.spthy").await;

    let res = s
        .client
        .get(s.url("/api/v1/theories/7/lemmas/Client_auth/initial-system"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 404);
    assert_eq!(
        res.json::<Value>().await.unwrap(),
        json!({"message": "Not Found"})
    );
    let res = s.client.get(s.url("/api/v1/nothing")).send().await.unwrap();
    assert_eq!(res.status(), 404);

    let res = s
        .client
        .get(lemma_url(&s, "no_such_lemma", "initial-system"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 400);
    assert_eq!(
        errors(&res.text().await.unwrap()),
        json!(["Lemma not found"])
    );

    let url = lemma_url(&s, "Client_auth", "execute");
    let (status, body) = post_form(&s, &url, &[("system", "x")]).await;
    assert_eq!(status, 400);
    assert_eq!(errors(&body), json!(["Missing method or system parameter"]));

    let sys = initial(&s, "Client_auth").await;
    let method = sys["encodedMethods"][0].as_str().unwrap();
    let token = sys["encodedSys"].as_str().unwrap();
    let (status, body) = post_form(&s, &url, &[("proofmethod", method), ("system", "AAAA")]).await;
    assert_eq!(status, 400);
    assert!(
        errors(&body)[0]
            .as_str()
            .unwrap()
            .starts_with("Failed to decode system: "),
        "{body}"
    );
    let (status, body) = post_form(&s, &url, &[("proofmethod", token), ("system", token)]).await;
    assert_eq!(status, 400);
    assert!(
        errors(&body)[0]
            .as_str()
            .unwrap()
            .starts_with("Failed to decode proof method: "),
        "{body}"
    );

    // A goal of a later system does not apply to the start system.
    let simplify = sys["proofMethods"]
        .as_array()
        .unwrap()
        .iter()
        .position(|m| m["name"] == "Simplify")
        .unwrap();
    let after = &execute(&s, "Client_auth", &sys, simplify).await[0]["system"];
    let goal = after["proofMethods"]
        .as_array()
        .unwrap()
        .iter()
        .position(|m| m["name"] == "SolveGoal")
        .unwrap();
    let goal = after["encodedMethods"][goal].as_str().unwrap();
    let (status, body) = post_form(&s, &url, &[("proofmethod", goal), ("system", token)]).await;
    assert_eq!(status, 400);
    assert_eq!(
        errors(&body),
        json!(["Proof method could not be applied to the given system"])
    );

    let url = lemma_url(&s, "Client_auth", "check-proof");
    let (status, body) = post_form(&s, &url, &[]).await;
    assert_eq!(status, 400);
    assert_eq!(errors(&body), json!(["Missing proof parameter"]));
    let (status, body) = post_form(&s, &url, &[("proof", "[1]")]).await;
    assert_eq!(status, 400);
    assert!(
        errors(&body)[0]
            .as_str()
            .unwrap()
            .starts_with("Invalid JSON in proof parameter: "),
        "{body}"
    );
    let (status, body) = post_form(&s, &url, &[("proof", r#"{"action": "solved"}"#)]).await;
    assert_eq!(status, 400);
    assert_eq!(
        errors(&body),
        json!(["Invalid proof JSON: missing or invalid 'children' field"])
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sapic_theories_are_refused() {
    if !maude_available() {
        return;
    }
    let s = start_server_with_theory("sapic_else_branch_exp.spthy").await;
    let res = s
        .client
        .get(lemma_url(&s, "any", "initial-system"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 400);
    assert_eq!(
        errors(&res.text().await.unwrap()),
        json!(["SAPIC theories are not supported by this API"])
    );
}

/// Follow the first method everywhere, as the autoprover does, and send the
/// proof back as the RL client does: a method token per step, `"solved"` /
/// `"contradictory"` at finished systems.
fn walk<'a>(
    s: &'a TestServer,
    lemma: &'a str,
    sys: Value,
    budget: &'a mut usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Value> + Send + 'a>> {
    Box::pin(async move {
        *budget -= 1;
        assert!(*budget > 0, "proof of {lemma} too long");
        match sys["status"].as_str().unwrap() {
            "Solved" => return json!({"action": "solved", "children": {}}),
            "Contradictory" => return json!({"action": "contradictory", "children": {}}),
            _ => {}
        }
        let cases = execute(s, lemma, &sys, 0).await;
        let mut children = serde_json::Map::new();
        for case in cases.as_array().unwrap() {
            let name = case["caseName"].as_str().unwrap().to_string();
            let child = walk(s, lemma, case["system"].clone(), budget).await;
            children.insert(name, child);
        }
        json!({"action": sys["encodedMethods"][0], "children": children})
    })
}

fn tree_size(tree: &Value) -> u64 {
    1 + tree["children"]
        .as_object()
        .unwrap()
        .values()
        .map(tree_size)
        .sum::<u64>()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_walked_proof_checks_as_complete() {
    if !maude_available() {
        return;
    }
    let s = start_server_with_theory("Tutorial.spthy").await;
    for lemma in ["Client_auth", "Client_session_key_secrecy"] {
        let start = initial(&s, lemma).await;
        let mut budget = 500;
        let tree = walk(&s, lemma, start, &mut budget).await;
        let proof = serde_json::to_string(&tree).unwrap();
        let (status, body) = post_form(
            &s,
            &lemma_url(&s, lemma, "check-proof"),
            &[("proof", &proof)],
        )
        .await;
        assert_eq!(status, 200, "{body}");
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["proofStatus"], "CompleteProof", "{lemma}");
        assert_eq!(
            body["proofSize"].as_u64().unwrap(),
            tree_size(&tree),
            "{lemma}"
        );
        let file = body["proofFile"].as_str().unwrap();
        assert!(file.contains(&format!("lemma {lemma}:")), "{file}");
        assert!(file.contains("by sorry /* removed */"), "{file}");
        assert!(!file.contains("/* unannotated */"), "{file}");

        // The fork also takes the tree as a JSON string holding the object.
        let wrapped = serde_json::to_string(&proof).unwrap();
        let (status, again) = post_form(
            &s,
            &lemma_url(&s, lemma, "check-proof"),
            &[("proof", &wrapped)],
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(serde_json::from_str::<Value>(&again).unwrap(), body);
    }
}

/// A proof 100 steps deep is 200 levels of JSON, beyond serde_json's default
/// limit of 128.  Only the first `Simplify` applies; the rest are reported
/// as an invalid step, as the fork does.
#[tokio::test(flavor = "multi_thread")]
async fn deep_proofs_are_accepted() {
    if !maude_available() {
        return;
    }
    let s = start_server_with_theory("Tutorial.spthy").await;
    let sys = initial(&s, "Client_auth").await;
    let simplify = sys["proofMethods"]
        .as_array()
        .unwrap()
        .iter()
        .position(|m| m["name"] == "Simplify")
        .unwrap();
    let mut tree = json!({"action": "contradictory", "children": {}});
    for _ in 0..100 {
        tree = json!({"action": sys["encodedMethods"][simplify], "children": {"": tree}});
    }
    let proof = serde_json::to_string(&tree).unwrap();
    let url = lemma_url(&s, "Client_auth", "check-proof");
    let (status, body) = post_form(&s, &url, &[("proof", &proof)]).await;
    assert_eq!(status, 200, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert!(
        body["proofFile"]
            .as_str()
            .unwrap()
            .contains("invalid proof step encountered"),
        "{body}"
    );
}

/// Parallel requests must give the same bytes as the same requests one at a
/// time: each runs on its own Maude process with its own fresh-variable
/// counter, so nothing one request does can leak into another.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_requests_match_sequential_ones() {
    if !maude_available() {
        return;
    }
    let s = std::sync::Arc::new(start_server_with_theory("Tutorial.spthy").await);
    let lemma = "Client_auth";

    // Concurrent first requests share one build of the lemma.
    let mut firsts = tokio::task::JoinSet::new();
    for _ in 0..6 {
        let s = s.clone();
        firsts.spawn(async move { initial(&s, lemma).await });
    }
    let start = initial(&s, lemma).await;
    while let Some(other) = firsts.join_next().await {
        assert_eq!(other.unwrap(), start);
    }

    // Requests from the first levels of the proof: every method at every
    // system, breadth first.
    const REQUESTS: usize = 40;
    let mut requests: Vec<(String, String)> = Vec::new();
    let mut queue = std::collections::VecDeque::from([start]);
    while let Some(sys) = queue.pop_front() {
        if requests.len() >= REQUESTS {
            break;
        }
        let n = sys["encodedMethods"].as_array().unwrap().len();
        for i in 0..n {
            requests.push((
                sys["encodedMethods"][i].as_str().unwrap().to_string(),
                sys["encodedSys"].as_str().unwrap().to_string(),
            ));
            let cases = execute(&s, lemma, &sys, i).await;
            queue.extend(
                cases
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c["system"].clone()),
            );
        }
    }
    assert!(requests.len() >= REQUESTS, "{}", requests.len());

    let url = lemma_url(&s, lemma, "execute");
    let mut sequential = BTreeMap::new();
    for (i, (method, sys)) in requests.iter().enumerate() {
        let form = [("proofmethod", method.as_str()), ("system", sys.as_str())];
        sequential.insert(i, post_form(&s, &url, &form).await);
    }
    let mut parallel = tokio::task::JoinSet::new();
    for (i, (method, sys)) in requests.into_iter().enumerate() {
        let (s, url) = (s.clone(), url.clone());
        parallel.spawn(async move {
            let form = [("proofmethod", method.as_str()), ("system", sys.as_str())];
            (i, post_form(&s, &url, &form).await)
        });
    }
    while let Some(done) = parallel.join_next().await {
        let (i, answer) = done.unwrap();
        assert_eq!(answer, sequential[&i], "request {i}");
    }
}
