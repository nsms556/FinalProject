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

// reactstrap components
import {Col, Container, Row} from "reactstrap";


const UserHeader = () => {

    const username = window.localStorage.getItem('username');

    return (
        <>
            <div
                className="header pb-8 pt-5 pt-lg-8 d-flex align-items-center"
                style={{
                    minHeight: "600px",
                    backgroundSize: "cover",
                    backgroundPosition: "center top",
                }}
            >
                {/* Mask */}
                <span className="mask bg-gradient-default opacity-8"/>
                {/* Header container */}
                <Container className="d-flex align-items-center" fluid>
                    <Row>
                        <Col>
                            <h1 className="display-2 text-white">Hello {username} ❣</h1>
                            <p className="text-white mt-0 mb-5">
                                지금까지 평가한 곡을 확인할 수 있습니다.<br/>
                                좋아하는 곡과 좋아하지 않는 곡을 편집할 수 있습니다.
                            </p>
                        </Col>
                    </Row>
                </Container>
            </div>
        </>
    );
};

export default UserHeader;
