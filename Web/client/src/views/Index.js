/*!

=========================================================
* Argon Dashboard React - v1.2.1
=========================================================

* Product Page: https://www.creative-tim.com/product/argon-dashboard-react
* Copyright 2021 Creative Tim (https://www.creative-tim.com)
* Licensed under MIT (https://github.com/creativetimofficial/argon-dashboard-react/blob/master/LICENSE.md)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/

import React from "react";

// reactstrap components
import {Card, CardBody, CardImg, CardText, CardTitle, Col, Container, Row} from "reactstrap";

import {GithubOutlined, HomeFilled} from "@ant-design/icons";

import Header from "components/Headers/Header.js";


const Index = () => {

    return (
        <>
            <Header/>
            <Container className="mt--7" fluid>
                <div style={{marginLeft: "auto", marginRight: "auto", width: "35%"}}>
                    <Card className="shadow" style={{padding: "1rem"}}>
                        <img alt="programmers" src="/images/mussg.gif"/>
                    </Card>
                </div>
                <br/>
                <Row>
                    <Col>
                        <Card style={{width: "18rem"}}>
                            <CardImg
                                alt="S"
                                src="/images/11ML.png"
                                top
                            />
                            <CardBody>
                                <CardTitle><strong>팀장</strong> S.</CardTitle>
                                <CardText>
                                    프로젝트 모델 개발 및 학습<br/>
                                    모델 성능 향상
                                </CardText>
                                <a href="https://github.com/AidevB6/FinalProject" target="_blank" rel="noreferrer">
                                    <HomeFilled/>
                                </a>
                                <span>　</span>
                                <a href="https://github.com/nsms556" target="_blank" rel="noreferrer">
                                    <GithubOutlined/>
                                </a>
                            </CardBody>
                        </Card>
                    </Col>
                    <Col>
                        <Card style={{width: "18rem"}}>
                            <CardImg
                                alt="L"
                                src="/images/15.png"
                                top
                            />
                            <CardBody>
                                <CardTitle>팀원 L.</CardTitle>
                                <CardText>
                                    데이터 전처리 및 모델 개발<br/>
                                    프로젝트 PPT 작성 및 발표
                                </CardText>
                                <a href="https://github.com/lymchgmk" target="_blank" rel="noreferrer">
                                    <GithubOutlined/>
                                </a>
                            </CardBody>
                        </Card>
                    </Col>
                    <Col>
                        <Card style={{width: "18rem"}}>
                            <CardImg
                                alt="B"
                                src="/images/27.png"
                                top
                            />
                            <CardBody>
                                <CardTitle>팀원 B.</CardTitle>
                                <CardText>
                                    Django framework 기반 API 개발<br/>
                                    DB 구축 및 AWS 서버 배포
                                </CardText>
                                <a href="https://github.com/spongebob03" target="_blank" rel="noreferrer">
                                    <GithubOutlined/>
                                </a>
                            </CardBody>
                        </Card>
                    </Col>
                    <Col>
                        <Card style={{width: "18rem"}}>
                            <CardImg
                                alt="Y"
                                src="/images/32.png"
                                top
                            />
                            <CardBody>
                                <CardTitle>팀원 Y.</CardTitle>
                                <CardText>
                                    웹 개발(FE)<br/>
                                    React 기반 웹 페이지 구축
                                </CardText>
                                <a href="https://github.com/devmei" target="_blank" rel="noreferrer">
                                    <GithubOutlined/>
                                </a>
                            </CardBody>
                        </Card>
                    </Col>
                </Row><br/>
            </Container>
        </>
    );
}

export default Index;
